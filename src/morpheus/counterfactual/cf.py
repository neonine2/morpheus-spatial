import copy
import sys
import _pickle as pickle
from typing import Callable, Optional, Tuple, Union
import numpy as np
import warnings

import torch
import torch.optim as optim

from ..api.defaults import DEFAULT_DATA, DEFAULT_META
from ..api.interfaces import Explainer, Explanation, FitMixin
from ..utils.gradients import perturb


class Counterfactual(Explainer, FitMixin):
    def __init__(
        self,
        predict: Union[Callable[[np.ndarray], np.ndarray], torch.nn.Module],
        input_transform: Union[Callable[[np.ndarray], np.ndarray], torch.nn.Module],
        shape: tuple,
        kappa: float = 0.0,
        beta: float = 0.1,
        feature_range: Tuple[Union[float, np.ndarray], Union[float, np.ndarray]] = (
            -1e10,
            1e10,
        ),
        gamma: float = 0.0,
        ae_model: Optional[torch.nn.Module] = None,
        enc_model: Optional[torch.nn.Module] = None,
        theta: float = 0.0,
        use_kdtree: bool = False,
        learning_rate_init: float = 1e-2,
        max_iterations: int = 1000,
        c_init: float = 10.0,
        c_steps: int = 10,
        eps: tuple = (1e-3, 1e-3),
        clip: tuple = (-1000.0, 1000.0),
        update_num_grad: int = 1,
        trustscore: Optional[str] = None,
        verbosity: int = 0,
        numerical_diff: bool = False,
        device: str = None,
    ) -> None:
        """
        Initialize prototypical counterfactual method.

        Parameters
        ----------
        predict
            `pytorch` model's prediction function returning class probabilities.
        shape
            Shape of input data starting with batch size.
        kappa
            Confidence parameter for the attack loss term.
        beta
            Regularization constant for L1 loss term.
        feature_range
            Tuple with `min` and `max` ranges to allow for perturbed instances. `Min` and `max` ranges can be `float`
            or `numpy` arrays with dimension (1x nb of features) for feature-wise ranges.
        gamma
            Regularization constant for optional auto-encoder loss term.
        ae_model
            Optional auto-encoder model used for loss regularization.
        enc_model
            Optional encoder model used to guide instance perturbations towards a class prototype.
        theta
            Constant for the prototype search loss term.
        use_kdtree
            Whether to use k-d trees for the prototype loss term if no encoder is available.
        learning_rate_init
            Initial learning rate of optimizer.
        max_iterations
            Maximum number of iterations for finding a counterfactual.
        c_init
            Initial value to scale the attack loss term.
        c_steps
            Number of iterations to adjust the constant scaling the attack loss term.
        eps
            If numerical gradients are used to compute `dL/dx = (dL/dp) * (dp/dx)`, then `eps[0]` is used to
            calculate `dL/dp` and `eps[1]` is used for `dp/dx`. `eps[0]` and `eps[1]` can be a combination of `float`
            values and `numpy` arrays. For `eps[0]`, the array dimension should be (1x nb of prediction categories)
            and for `eps[1]` it should be (1x nb of features).
        clip
            Tuple with min and max clip ranges for both the numerical gradients and the gradients
            obtained from the `tensorflow` graph.
        update_num_grad
            If numerical gradients are used, they will be updated every `update_num_grad` iterations.
        trustscore
            Directory where trustscore object is to be used
        sess
            Optional `tensorflow` session that will be used if passed instead of creating or inferring one internally.
        """
        super().__init__(meta=copy.deepcopy(DEFAULT_META))

        params = locals()
        remove = [
            "self",
            "predict",
            "input_transform",
            "ae_model",
            "enc_model",
            "__class__",
        ]
        for key in remove:
            params.pop(key)
        self.meta["params"].update(params)
        self.predict = predict
        self.trustscore = trustscore
        self.numerical_diff = numerical_diff
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # check if predict is a pytorch model
        is_model = True
        if is_model:
            self.classes = self.predict(torch.zeros(size=shape)).shape[1]
        else:
            self.numerical_diff = True
            self.classes = self.predict.predict(np.zeros(shape)).shape[1]

        is_ae = isinstance(ae_model, torch.nn.Module)
        is_enc = isinstance(enc_model, torch.nn.Module)

        if is_enc:
            self.enc_model = True
        else:
            self.enc_model = False

        if is_ae:
            self.ae_model = True
        else:
            self.ae_model = False

        if use_kdtree and self.enc_model:
            warnings.warn(
                "Both an encoder and k-d trees enabled. Using the encoder for the prototype loss term."
            )

        if use_kdtree or self.enc_model:
            self.enc_or_kdtree = True
        else:
            self.enc_or_kdtree = False
        self.meta["params"].update(enc_or_kdtree=self.enc_or_kdtree)

        self.shape = shape
        self.kappa = kappa
        self.beta = beta
        self.gamma = gamma
        self.theta = theta
        self.ae = ae_model
        self.enc = enc_model
        self.use_kdtree = use_kdtree
        self.batch_size = shape[0]
        self.max_iterations = max_iterations
        self.c_init = torch.tensor(c_init)
        self.c_steps = c_steps
        self.feature_range = tuple(
            [
                (
                    (torch.ones(shape[1:]) * feature_range[_])[None, :]
                    if isinstance(feature_range[_], float)
                    else feature_range[_]
                )
                for _ in range(2)
            ]
        )
        self.update_num_grad = update_num_grad
        self.eps = eps
        self.clip = clip
        self.input_transform = input_transform
        self.learning_rate_init = learning_rate_init
        self.verbosity = verbosity

        # variable for target class proto
        if self.enc_model:
            self.shape_enc = self.enc.predict(np.zeros(self.shape)).shape  # type: ignore[union-attr]
        else:
            # shape_env may not be equal to shape
            self.shape_enc = self.shape

        # Use the custom function
        warnings.showwarning = simple_warning

    def compute_shrinkage_thresholding(self, feature_range):
        # Conditions for element-wise shrinkage thresholding
        cond_0 = (self.adv_s - self.orig) > self.beta
        cond_1 = torch.abs(self.adv_s - self.orig) <= self.beta
        cond_2 = (self.adv_s - self.orig) < -self.beta

        # Calculate values based on conditions
        upper = torch.min(self.adv_s - self.beta, feature_range[1])
        lower = torch.max(self.adv_s + self.beta, feature_range[0])

        # Compute the final tensor based on conditions
        self.assign_adv = (
            cond_0.float() * upper + cond_1.float() * self.orig + cond_2.float() * lower
        )

    def compute_perturbation_projection(self):
        # perturbation update (momentum term)
        self.zt = self.global_step / (self.global_step + 3.0)
        self.assign_adv_s = self.assign_adv + self.zt * (self.assign_adv - self.adv)

        # vector projection on correct feature range set
        self.assign_adv_s = torch.clamp(
            self.assign_adv_s, self.feature_range[0], self.feature_range[1]
        )

    def update_counterfactuals(self):
        # Update counterfactuals of step k+1 to k
        self.adv = self.assign_adv.clone()
        self.adv_s.data = self.assign_adv_s.data.clone()

    def compute_deviations(self):
        # Derive deviations
        self.delta = self.orig - self.adv
        self.delta_s = self.orig - self.adv_s

    def compute_l1_l2_losses(self, shape):
        # Compute L1 and L2 losses
        self.compute_deviations()

        ax_sum = tuple(range(1, len(shape)))

        self.l2 = torch.sum(self.delta**2, dim=ax_sum)
        self.l2_s = torch.sum(self.delta_s**2, dim=ax_sum)

        self.l1 = torch.sum(torch.abs(self.delta), dim=ax_sum)
        self.l1_s = torch.sum(torch.abs(self.delta_s), dim=ax_sum)

        self.l1_l2 = self.l2 + self.l1 * self.beta
        self.l1_l2_s = self.l2_s + self.l1_s * self.beta

        # Sum losses
        self.loss_l1 = torch.sum(self.l1)
        self.loss_l1_s = torch.sum(self.l1_s)
        self.loss_l2 = torch.sum(self.l2)
        self.loss_l2_s = torch.sum(self.l2_s)

    def compute_autoencoder_loss(self):
        # Autoencoder loss
        if self.ae_model:
            # Run autoencoder
            self.adv_ae = self.ae(self.adv)
            self.adv_ae_s = self.ae(self.adv_s)
            # Compute loss
            self.loss_ae = self.gamma * torch.norm(self.adv_ae - self.adv) ** 2
            self.loss_ae_s = self.gamma * torch.norm(self.adv_ae_s - self.adv_s) ** 2
        else:  # No auto-encoder available
            self.loss_ae = torch.tensor(0.0)
            self.loss_ae_s = torch.tensor(0.0)

    def compute_attack_loss(self):
        self.pred_proba = self.predict(self.adv)
        self.pred_proba_s = self.predict(self.adv_s)

        if self.numerical_diff:
            with torch.no_grad():
                self.loss_attack = self.loss_fn(self.pred_proba, self.target)
        elif self.c_init == 0.0 and self.c_steps == 1:  # Prediction loss term not used
            self.loss_attack = torch.tensor(0.0)
            self.loss_attack_s = torch.tensor(0.0)
        else:
            # Probability of target label prediction
            self.target_proba = torch.sum(self.target * self.pred_proba, dim=1)
            target_proba_s = torch.sum(self.target * self.pred_proba_s, dim=1)

            # Max probability of non-target label prediction
            self.nontarget_proba_max = torch.max(
                (1 - self.target) * self.pred_proba - (self.target * 10000), dim=1
            )[0]
            nontarget_proba_max_s = torch.max(
                (1 - self.target) * self.pred_proba_s - (self.target * 10000), dim=1
            )[0]

            # Loss term f(x,d)
            loss_attack = torch.maximum(
                torch.tensor(0.0),
                -self.nontarget_proba_max + self.target_proba + self.kappa,
            )
            loss_attack_s = torch.maximum(
                torch.tensor(0.0), -nontarget_proba_max_s + target_proba_s + self.kappa
            )

            # c * f(x,d)
            self.loss_attack = torch.sum(self.const * loss_attack)
            self.loss_attack_s = torch.sum(self.const * loss_attack_s)

    def compute_prototype_loss(self):
        """Compute the prototype loss."""
        if self.enc_model:
            self.loss_proto = (
                self.theta * torch.norm(self.enc(self.adv) - self.target_proto) ** 2
            )
            self.loss_proto_s = (
                self.theta * torch.norm(self.enc(self.adv_s) - self.target_proto) ** 2
            )
        elif self.use_kdtree:
            self.loss_proto = self.theta * torch.norm(self.adv - self.target_proto) ** 2
            self.loss_proto_s = (
                self.theta * torch.norm(self.adv_s - self.target_proto) ** 2
            )
        else:  # No encoder available and no k-d trees used
            self.loss_proto = torch.tensor(0.0)
            self.loss_proto_s = torch.tensor(0.0)

    def compute_regularizer_loss(self):

        # Computer each loss term
        self.compute_l1_l2_losses(self.shape)
        self.compute_autoencoder_loss()
        self.compute_prototype_loss()

        """Compute the regularization loss."""
        self.loss_reg = self.loss_l2_s + self.loss_ae_s + self.loss_proto_s

        # add L1 term to overall loss; this is not the loss that will be directly optimized
        self.loss_total = (
            self.loss_attack
            + self.loss_l2
            + self.loss_ae
            + self.beta * self.loss_l1
            + self.loss_proto
        )

    def loss_fn(self, pred_proba: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Compute the attack loss.

        Parameters
        ----------
        pred_proba
            Prediction probabilities of an instance.
        Y
            One-hot representation of instance labels.

        Returns
        -------
        Loss of the attack.
        """
        # probability of target label prediction
        target_proba = torch.sum(pred_proba * Y)
        # max probability of non target label prediction
        nontarget_proba_max = torch.max((1 - Y) * pred_proba - 10000 * Y)
        # loss term f(x,d)
        loss = torch.maximum(
            torch.tensor([0.0]), -nontarget_proba_max + target_proba + self.kappa
        )
        # c * f(x,d)
        loss_attack = torch.sum(self.const * loss)
        return loss_attack

    def setup_training(self, learning_rate_init):
        """Set up the training parameters."""
        self.optimizer = optim.SGD([self.adv_s], lr=learning_rate_init)
        lambda_poly_decay = lambda epoch: (1 - epoch / self.max_iterations) ** 0.5
        self.learning_rate = optim.lr_scheduler.LambdaLR(
            optimizer=self.optimizer, lr_lambda=lambda_poly_decay
        )

    def fit(
        self,
        train_data: np.ndarray = np.array([]),
        preds: np.ndarray = np.array([]),
    ) -> "Counterfactual":
        """
        Get prototypes for each class using the encoder or k-d trees.
        The prototypes are used for the encoder loss term or to calculate the optional trust scores.

        Parameters
        ----------
        train_data
            Representative sample from the training data.
        """
        # get params for storage in meta
        params = locals()
        remove = ["self", "train_data"]
        for key in remove:
            params.pop(key)
        # update metadata
        self.meta["params"].update(params)

        if self.enc_model:
            enc_data = self.enc.predict(train_data)  # type: ignore[union-attr]
            self.class_proto = {}  # type: dict
            self.class_enc = {}  # type: dict
            for i in range(self.classes):
                idx = np.where(preds == i)[0]
                self.class_proto[i] = np.expand_dims(
                    np.mean(enc_data[idx], axis=0), axis=0
                )
                self.class_enc[i] = enc_data[idx]
        elif self.use_kdtree:
            # warnings.warn(
            #     "No encoder specified. Using k-d trees to represent class prototypes."
            # )
            ts = load_object(self.trustscore)
            self.kdtrees = ts.kdtrees
            self.X_by_class = ts.X_kdtree
        return self

    def get_gradients(
        self, X: np.ndarray, Y: np.ndarray, grads_shape: tuple
    ) -> np.ndarray:
        """
        Compute numerical gradients of the attack loss term:
        `dL/dx = (dL/dP)*(dP/dx)` with `L = loss_attack_s; P = predict; x = adv_s`.

        Parameters
        ----------
        X
            Instance around which gradient is evaluated.
        Y
            One-hot representation of instance labels.
        grads_shape
            Shape of gradients.

        Returns
        -------
        Array with gradients.
        """
        X_pred = X.clone().detach()
        # N = gradient batch size; F = nb of features; P = nb of prediction classes; B = instance batch size
        # dL/dP -> BxP
        preds = self.predict(X_pred).detach().numpy()  # NxP
        preds_pert_pos, preds_pert_neg = perturb(
            preds, self.eps[0], proba=True
        )  # (N*P)xP

        def f(preds_pert):
            return np.sum(Y * preds_pert, axis=1)

        def g(preds_pert):
            return np.max((1 - Y) * preds_pert, axis=1)

        # find instances where the gradient is 0
        idx_nograd = np.where(f(preds) - g(preds) <= -self.kappa)[0]
        if len(idx_nograd) == X.shape[0]:
            return np.zeros((1, *X.shape[1:]))
        dl_df = f(preds_pert_pos) - f(preds_pert_neg)  # N*P
        dl_dg = g(preds_pert_pos) - g(preds_pert_neg)  # N*P
        dl_dp = dl_df - dl_dg  # N*P
        dl_dp = np.reshape(dl_dp, (X.shape[0], -1)) / (2 * self.eps[0])  # NxP

        # dP/dx -> PxF
        X_pert_pos, X_pert_neg = perturb(
            X, self.eps[1], proba=False
        )  # (N*F)x(shape of X[0])
        X_pert = np.concatenate([X_pert_pos, X_pert_neg], axis=0)
        preds_concat = self.predict(torch.tensor(X_pert)).detach().numpy()  # (N*F)*P
        n_pert = X_pert_pos.shape[0]
        dp_dx = preds_concat[:n_pert] - preds_concat[n_pert:]  # (N*F)*P
        dp_dx = np.reshape(
            np.reshape(dp_dx, (X.shape[0], -1)),
            (X.shape[0], preds.shape[1], -1),
            order="F",
        ) / (
            2 * self.eps[1]
        )  # NxPxF

        # dL/dx -> Bx(shape of X[0])
        grads = np.einsum("ij,ijk->ik", dl_dp, dp_dx)  # NxF
        # set instances where gradient is 0 to 0
        if len(idx_nograd) > 0:
            grads[idx_nograd] = np.zeros(grads.shape[1:])
        grads = np.mean(grads, axis=0)  # B*F
        grads = np.reshape(grads, (self.batch_size,) + grads_shape)  # B*(shape of X[0])
        return grads

    def score(
        self, X: np.ndarray, adv_class: int, orig_class: int, eps: float = 1e-10
    ) -> float:
        """
        Parameters
        ----------
        X
            Perturbation.
        adv_class
            Predicted class on the perturbed instance.
        orig_class
            Predicted class on the original instance.
        eps
            Small number to avoid dividing by 0.

        Returns
        -------
        Ratio between the distance to the prototype of the predicted class for the original instance and \
        the prototype of the predicted class for the perturbed instance.
        """
        if self.enc_model:
            X_enc = self.enc.predict(X)  # type: ignore[union-attr]
            adv_proto = self.class_proto[adv_class]
            orig_proto = self.class_proto[orig_class]
            dist_adv = np.linalg.norm(X_enc - adv_proto)
            dist_orig = np.linalg.norm(X_enc - orig_proto)
        elif self.use_kdtree:
            # Xpatch = self.input_transform(X).eval(session=self.sess)
            # Xpatch = Xpatch.reshape(Xpatch.shape[0], -1)
            dist_adv = self.kdtrees[adv_class].query(Xpatch, k=1)[0]
            dist_orig = self.kdtrees[orig_class].query(Xpatch, k=1)[0]
        else:
            warnings.warn(
                "Need either an encoder or the k-d trees enabled to compute distance scores."
            )
        return dist_orig / (dist_adv + eps)  # type: ignore[return-value]

    def attack(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        target_class: Optional[list] = None,
        k: Optional[int] = None,
        k_type: str = "mean",
        threshold: float = 0.0,
        print_every: int = 100,
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Find a counterfactual (CF) for instance `X` using a fast iterative shrinkage-thresholding algorithm (FISTA).

        Parameters mk
        ----------
        X
            Instance to attack.
        Y
            Labels for `X` as one-hot-encoding.
        target_class
            List with target classes used to find closest prototype. If ``None``, the nearest prototype
            except for the predict class on the instance is used.
        k
            Number of nearest instances used to define the prototype for a class. Defaults to using all
            instances belonging to the class if an encoder is used and to 1 for k-d trees.
        k_type
            Use either the average encoding of the k nearest instances in a class (``k_type='mean'``) or
            the k-nearest encoding in the class (``k_type='point'``) to define the prototype of that class.
            Only relevant if an encoder is used to define the prototypes.
        threshold
            Threshold level for the ratio between the distance of the counterfactual to the prototype of the
            predicted class for the original instance over the distance to the prototype of the predicted class
            for the counterfactual. If the trust score is below the threshold, the proposed counterfactual does
            not meet the requirements.
        verbosity
            Print intermediate results of optimization if greater than 1.
        print_every
            Print frequency if verbose is greater than 1.
        log_every
            `tensorboard` log frequency if write directory is specified.

        Returns
        -------
        Overall best attack and gradients for that attack.
        """
        # make sure nb of instances in X equals batch size
        assert self.batch_size == X.shape[0]

        def compare(x: Union[float, int, np.ndarray], y: int) -> bool:
            """
            Compare predictions with target labels and return whether counterfactual conditions hold.

            Parameters
            ----------
            x
                Predicted class probabilities or labels.
            y
                Target or predicted labels.

            Returns
            -------
            Bool whether counterfactual conditions hold.
            """
            if (not isinstance(x, (float, int, np.int64))) and (
                not is_single_float_or_int_tensor(x)
            ):
                x = x.clone()
                x[y] += self.kappa  # type: ignore
                x = torch.argmax(x)  # type: ignore
            return x != y

        # define target classes for prototype if not specified yet
        if target_class is None:
            target_class = list(range(self.classes))
            target_class.remove(np.argmax(Y, axis=1))
            if self.verbosity > 1:
                print("Predicted class: {}".format(np.argmax(Y, axis=1)))
                print("Target classes: {}".format(target_class))
        X_num = X

        # find closest prototype in the target class list
        dist_proto = {}
        if self.enc_model:

            X_enc = self.enc.predict(self.input_transform(X))  # type: ignore[union-attr]
            class_dict = self.class_proto if k is None else self.class_enc

            for c, v in class_dict.items():
                if c not in target_class:
                    continue
                if k is None:
                    dist_proto[c] = np.linalg.norm(X_enc - v)
                elif k is not None:
                    dist_k = np.linalg.norm(
                        X_enc.reshape(X_enc.shape[0], -1) - v.reshape(v.shape[0], -1),
                        axis=1,
                    )
                    idx = np.argsort(dist_k)[:k]
                    if k_type == "mean":
                        dist_proto[c] = np.mean(dist_k[idx])
                    else:
                        dist_proto[c] = dist_k[idx[-1]]
                    self.class_proto[c] = np.expand_dims(
                        np.mean(v[idx], axis=0), axis=0
                    )
        elif self.use_kdtree:
            if k is None:
                k = 1
            self.class_proto = {}
            for c in range(self.classes):
                if c not in target_class:
                    continue
                dist_c, idx_c = self.kdtrees[c].query(X_num.cpu(), k=k)
                dist_proto[c] = dist_c[0][-1]
                self.class_proto[c] = self.X_by_class[c][idx_c[0][-1]].reshape(1, -1)

        if self.enc_or_kdtree:
            self.id_proto = min(dist_proto, key=dist_proto.get)
            proto_val = self.class_proto[self.id_proto].to(self.device)
            if self.verbosity > 1:
                print("Prototype class: {}".format(self.id_proto))
        else:  # no prototype loss term used
            torch.zeros(self.shape_enc)

        # set shape for perturbed instance and gradients
        pert_shape = self.shape

        # set the lower and upper bounds for the constant 'c' to scale the attack loss term
        # these bounds are updated for each c_step iteration
        const_lb = torch.zeros(self.batch_size)
        const = torch.ones(self.batch_size) * self.c_init
        const_ub = torch.ones(self.batch_size) * 1e10

        # init values for the best attack instances for each instance in the batch
        overall_best_dist = [1e10] * self.batch_size
        overall_best_attack = [torch.zeros(self.shape[1:])] * self.batch_size
        overall_best_grad = (torch.zeros(self.shape), torch.zeros(self.shape))

        # keep track of counterfactual evolution
        self.cf_global = {i: [] for i in range(self.c_steps)}  # type: dict

        # iterate over nb of updates for 'c'
        for _ in range(self.c_steps):

            # init variables
            self.orig = X_num.clone().detach()
            self.target = Y.clone().detach()
            self.const = const.clone().detach()
            self.adv = X_num.clone().detach()
            self.adv_s = X_num.clone().requires_grad_()
            self.target_proto = proto_val.clone().detach()

            # init optimizer
            self.setup_training(self.learning_rate_init)

            # reset current best distances and scores
            current_best_dist = [1e10] * self.batch_size
            current_best_proba = [-1] * self.batch_size
            self.global_step = 0

            X_der_batch, X_der_batch_s = [], []  # type: Any, Any

            # Zero out the gradients before computation at every step
            self.optimizer.zero_grad()

            # Compute the forward loss
            self.compute_attack_loss()
            self.compute_regularizer_loss()

            for i in range(self.max_iterations):

                # numerical gradients
                grads_num = np.zeros(pert_shape)
                grads_num_s = np.zeros(pert_shape)

                # check if numerical gradient computation is needed
                if self.numerical_diff and (self.c_init != 0.0 or self.c_steps > 1):
                    X_der = self.adv.clone().detach()
                    X_der_s = self.adv_s.clone().detach()
                    Y_numpy = Y.clone().detach().numpy()

                    X_der_batch.append(X_der)
                    X_der_batch_s.append(X_der_s)
                    if (
                        i % self.update_num_grad == 0 and i > 0
                    ):  # compute numerical gradients
                        c = self.const.numpy()
                        X_der_batch = torch.cat(X_der_batch)
                        X_der_batch_s = torch.cat(X_der_batch_s)
                        grads_num = (
                            self.get_gradients(
                                X_der_batch, Y_numpy, grads_shape=pert_shape[1:]
                            )
                            * c
                        )
                        grads_num_s = (
                            self.get_gradients(
                                X_der_batch_s, Y_numpy, grads_shape=pert_shape[1:]
                            )
                            * c
                        )
                        # clip gradients
                        grads_num = np.clip(grads_num, self.clip[0], self.clip[1])  # type: ignore
                        grads_num_s = np.clip(grads_num_s, self.clip[0], self.clip[1])  # type: ignore
                        X_der_batch, X_der_batch_s = [], []
                elif not self.numerical_diff and i > 0:
                    self.loss_attack_s.backward()
                    with torch.no_grad():
                        self.adv_s.grad.clamp_(self.clip[0], self.clip[1])

                self.loss_reg.backward(retain_graph=True)
                with torch.no_grad():
                    if self.numerical_diff or i == 0:
                        self.adv_s.grad.clamp_(self.clip[0], self.clip[1])
                    else:
                        self.adv_s.grad.clamp_(self.clip[0] * 2, self.clip[1] * 2)
                    self.adv_s.grad.add_(torch.from_numpy(grads_num_s))
                    gradients = self.adv_s.grad

                # Apply the gradients
                self.optimizer.step()

                # Update the learning rate
                self.learning_rate.step()

                # Increment the global step
                self.global_step += 1

                # update adv and adv_s
                with torch.no_grad():
                    self.compute_shrinkage_thresholding(self.feature_range)
                    self.compute_perturbation_projection()
                    self.update_counterfactuals()

                # compute overall and attack loss, L1+L2 loss, prediction probabilities
                # on perturbed instances and new adv
                # L1+L2 and prediction probabilities used to see if adv is better than the current best adv under FISTA
                self.optimizer.zero_grad()

                self.compute_attack_loss()
                self.compute_regularizer_loss()

                with torch.no_grad():
                    if self.verbosity > 1 and i % print_every == 0:

                        target_proba = torch.sum(self.pred_proba * Y)
                        nontarget_proba_max = torch.max((1 - Y) * self.pred_proba)

                        print("\nIteration: {}; Const: {}".format(i, const[0]))
                        print(
                            "Loss total: {:.3f}, loss attack: {:.3f}".format(
                                self.loss_total, self.loss_attack
                            )
                        )
                        print(
                            "L2: {:.3f}, L1: {:.3f}".format(self.loss_l2, self.loss_l1)
                        )
                        print("Loss proto: {:.3f}".format(self.loss_proto))
                        print(
                            "Target proba: {:.2f}, max non target proba: {:.2f}".format(
                                target_proba, nontarget_proba_max
                            )
                        )
                        # Ensure gradients are not None
                        if gradients is not None:
                            # Convert gradients to a numpy array for printing if necessary
                            gradients_np = gradients.detach().cpu().numpy()  # type: ignore
                            print(
                                "Gradient min/max: {:.3f}/{:.3f}".format(
                                    gradients_np.min(), gradients_np.max()
                                )
                            )
                            print("Gradient mean/abs mean: {:.3f}/{:.3f}".format(gradients_np.mean(), np.mean(np.abs(gradients_np))))  # type: ignore
                        else:
                            print("Gradient min/max: None")
                            print("Gradient mean/abs mean: None")
                        if self.numerical_diff:
                            print(
                                "Gradient numerical attack min/max: {:.3f}/{:.3f}".format(
                                    grads_num.min(), grads_num.max()
                                )
                            )  # type: ignore
                            print(
                                "Gradient numerical mean/abs mean: {:.3f}/{:.3f}".format(
                                    np.mean(grads_num), np.mean(np.abs(grads_num))
                                )
                            )  # type: ignore
                        sys.stdout.flush()

                    # update best perturbation (distance) and class probabilities
                    # if beta * L1 + L2 < current best and predicted label is different from the initial label:
                    # update best current step or global perturbations
                    for batch_idx, (dist, proba, adv_idx) in enumerate(
                        zip(self.l1_l2, self.pred_proba, self.adv)
                    ):
                        Y_class = torch.argmax(Y[batch_idx])
                        adv_class = torch.argmax(proba)
                        adv_idx = adv_idx.unsqueeze(0)

                        # calculate trust score
                        if threshold > 0.0:
                            score = self.score(adv_idx, np.argmax(pred_proba), Y_class)  # type: ignore
                            above_threshold = score > threshold
                        else:
                            above_threshold = True

                        # current step
                        if (
                            dist < current_best_dist[batch_idx]
                            and compare(proba, Y_class)  # type: ignore
                            and above_threshold
                            and adv_class in target_class
                        ):
                            current_best_dist[batch_idx] = dist
                            current_best_proba[batch_idx] = adv_class  # type: ignore

                        # global
                        if (
                            dist < overall_best_dist[batch_idx]
                            and compare(proba, Y_class)  # type: ignore
                            and above_threshold
                            and adv_class in target_class
                        ):
                            if self.verbosity > 1:
                                print("\nNew best counterfactual found!")
                            overall_best_dist[batch_idx] = dist
                            overall_best_attack[batch_idx] = adv_idx
                            overall_best_grad = gradients
                            self.best_attack = True
                            self.cf_global[_].append(adv_idx)

            # adjust the 'c' constant for the first loss term
            for batch_idx in range(self.batch_size):
                if (
                    compare(
                        current_best_proba[batch_idx], np.argmax(Y[batch_idx].cpu())
                    )
                    and current_best_proba[batch_idx] != -1  # type: ignore
                ):
                    # want to refine the current best solution by putting more emphasis on the regularization terms
                    # of the loss by reducing 'c'; aiming to find a perturbation closer to the original instance
                    const_ub[batch_idx] = min(const_ub[batch_idx], const[batch_idx])
                    if const_ub[batch_idx] < 1e9:
                        const[batch_idx] = (
                            const_lb[batch_idx] + const_ub[batch_idx]
                        ) / 2
                else:
                    # no valid current solution; put more weight on the first loss term to try and meet the
                    # prediction constraint before finetuning the solution with the regularization terms
                    const_lb[batch_idx] = max(
                        const_lb[batch_idx], const[batch_idx]
                    )  # update lower bound to constant
                    if const_ub[batch_idx] < 1e9:
                        const[batch_idx] = (
                            const_lb[batch_idx] + const_ub[batch_idx]
                        ) / 2
                    else:
                        const[batch_idx] *= 10

        # return best overall attack
        best_attack = np.concatenate(
            [item.cpu() for item in overall_best_attack], axis=0
        )
        if best_attack.shape != self.shape:
            best_attack = np.expand_dims(best_attack, axis=0)

        return best_attack, overall_best_grad

    def explain(
        self,
        # patch: np.ndarray,
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
        target_class: Optional[list] = None,
        k: Optional[int] = None,
        k_type: str = "mean",
        threshold: float = 0.0,
        print_every: int = 100,
    ) -> Explanation:
        """
        Explain instance and return counterfactual with metadata.

        Parameters
        ----------
        X
            Initial perturbation
        Y
            Labels for `X` as one-hot-encoding.
        target_class
            List with target classes used to find closest prototype. If ``None``, the nearest prototype
            except for the predict class on the instance is used.
        k
            Number of nearest instances used to define the prototype for a class. Defaults to using all
            instances belonging to the class if an encoder is used and to 1 for k-d trees.
        k_type
            Use either the average encoding of the `k` nearest instances in a class (``k_type='mean'``) or
            the k-nearest encoding in the class (``k_type='point'``) to define the prototype of that class.
            Only relevant if an encoder is used to define the prototypes.
        threshold
            Threshold level for the ratio between the distance of the counterfactual to the prototype of the
            predicted class for the original instance over the distance to the prototype of the predicted class
            for the counterfactual. If the trust score is below the threshold, the proposed counterfactual does
            not meet the requirements.
        verbosity
            Print intermediate results of optimization if greater than 1.
        print_every
            Print frequency if verbosity is greater than 1.
        log_every
            `tensorboard` log frequency if write directory is specified

        Returns
        -------
        explanation
            `Explanation` object containing the counterfactual with additional metadata as attributes.
        """
        if X.shape[0] != 1:
            warnings.warn(
                "Currently only single instance explanations supported (first dim = 1), "
                "but first dim = %s",
                X.shape[0],
            )

        # output explanation dictionary
        data = copy.deepcopy(DEFAULT_DATA)

        # add original prediction to explanation
        Y_proba = self.predict(X).detach().cpu().numpy()
        Y_ohe = np.zeros(Y_proba.shape)
        Y_class = np.argmax(Y_proba, axis=1)
        Y_ohe[np.arange(Y_proba.shape[0]), Y_class] = 1
        Y = torch.from_numpy(Y_ohe).clone().to(self.device)

        data["orig_proba"] = Y_proba
        data["orig_class"] = np.argmax(Y_proba, axis=1)[0]

        # find best counterfactual
        self.best_attack = False  # flag to indicate whether a CF was found
        best_attack, grads = self.attack(
            X,
            Y=Y,
            target_class=target_class,
            k=k,
            k_type=k_type,
            threshold=threshold,
            print_every=print_every,
        )

        # add id of prototype to explanation dict
        if self.enc_or_kdtree:
            data["id_proto"] = self.id_proto

        # add to explanation dict
        if not self.best_attack:
            if self.verbosity > 0:
                warnings.warn("No counterfactual found!")

            # create explanation object
            explanation = Explanation(meta=copy.deepcopy(self.meta), data=data)
            return explanation

        data["all"] = self.cf_global
        data["cf"] = {}
        data["cf"]["X"] = best_attack
        Y_pert = (
            self.predict(torch.from_numpy(best_attack).to(self.device))
            .detach()
            .cpu()
            .numpy()
        )
        data["cf"]["proba"] = Y_pert
        data["cf"]["class"] = np.argmax(Y_pert, axis=1)[0]
        data["cf"]["grads"] = grads.detach().cpu().numpy()

        # create explanation object
        explanation = Explanation(meta=copy.deepcopy(self.meta), data=data)

        return explanation


def load_object(filename):
    with open(filename, "rb") as outp:
        return pickle.load(outp)


def is_single_float_or_int_tensor(x):
    # Check if x is a tensor and has only one element
    if isinstance(x, torch.Tensor) and x.numel() == 1:
        # Check for integer types (can include more types if needed)
        if x.dtype in [torch.int32, torch.int64, torch.int16, torch.int8]:
            return True
        # Check for float types (can include more types if needed)
        elif x.dtype in [torch.float32, torch.float64, torch.float16]:
            return True
        # Add more conditions if you want to check for other specific types
    return False


# Define a custom warning function
def simple_warning(message, category, filename, lineno, file=None, line=None):
    print(f"{category.__name__}: {message}")
