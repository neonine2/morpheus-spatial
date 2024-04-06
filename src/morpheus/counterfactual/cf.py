import os
import copy
import logging
import sys
import _pickle as pickle
from typing import Any, Callable, Optional, Tuple, Union
import numpy as np
import tensorflow.compat.v1 as tf

from ..api.defaults import DEFAULT_DATA, DEFAULT_META
from ..api.interfaces import Explainer, Explanation, FitMixin
from ..utils.gradients import perturb
import torch

logger = logging.getLogger(__name__)
tf.logging.set_verbosity(tf.logging.ERROR)

# disable eager execution
tf.compat.v1.disable_eager_execution()


class Counterfactual(Explainer, FitMixin):

    def __init__(
        self,
        predict: Union[Callable[[np.ndarray], np.ndarray], tf.keras.Model],
        input_transform: Union[Callable[[np.ndarray], np.ndarray], tf.keras.Model],
        shape: tuple,
        kappa: float = 0.0,
        beta: float = 0.1,
        feature_range: Tuple[Union[float, np.ndarray], Union[float, np.ndarray]] = (
            -1e10,
            1e10,
        ),
        gamma: float = 0.0,
        ae_model: Optional[tf.keras.Model] = None,
        enc_model: Optional[tf.keras.Model] = None,
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
        write_dir: Optional[str] = None,
        sess: Optional[tf.Session] = None,
        verbose: bool = False,
    ) -> None:
        """
        Initialize prototypical counterfactual method.

        Parameters
        ----------
        predict
            `tensorflow` model or any other model's prediction function returning class probabilities.
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
        write_dir
            Directory to write `tensorboard` files to.
        sess
            Optional `tensorflow` session that will be used if passed instead of creating or inferring one internally.
        """
        super().__init__(meta=copy.deepcopy(DEFAULT_META))

        # if image as input
        if len(shape) > 2:
            patch_shape = shape[
                1:
            ]  # should be [x length, y length, and number of color channels]
            shape = (shape[0], shape[-1])
        else:
            patch_shape = shape
        params = locals()
        remove = [
            "self",
            "predict",
            "input_transform",
            "ae_model",
            "enc_model",
            "sess",
            "__class__",
        ]
        for key in remove:
            params.pop(key)
        self.meta["params"].update(params)

        self.predict = predict
        self.trustscore = trustscore

        # check if the passed object is a model and get session
        is_model = isinstance(predict, tf.keras.Model)
        model_sess = tf.compat.v1.keras.backend.get_session()
        is_ae = isinstance(ae_model, tf.keras.Model)
        is_enc = isinstance(enc_model, tf.keras.Model)
        self.meta["params"].update(is_model=is_model, is_ae=is_ae, is_enc=is_enc)

        # if session provided, use it
        if isinstance(sess, tf.Session):
            self.sess = sess
        else:
            self.sess = model_sess

        if is_model:  # Keras or TF model
            self.model = True
            self.classes = self.predict.predict(np.zeros(shape)).shape[1]  # type: ignore
        else:  # black-box model
            self.model = False
            self.classes = self.predict(torch.zeros(size=shape)).shape[1]
            print(f"Black box model with {self.classes} classes")

        if is_enc:
            self.enc_model = True
        else:
            self.enc_model = False

        if is_ae:
            self.ae_model = True
        else:
            self.ae_model = False

        if use_kdtree and self.enc_model:
            logger.warning(
                "Both an encoder and k-d trees enabled. Using the encoder for the prototype loss term."
            )

        if use_kdtree or self.enc_model:
            self.enc_or_kdtree = True
        else:
            self.enc_or_kdtree = False
        self.meta["params"].update(enc_or_kdtree=self.enc_or_kdtree)

        self.meta["params"].update(is_cat=False)

        self.shape = shape
        self.patch_shape = patch_shape
        self.kappa = kappa
        self.beta = beta
        self.gamma = gamma
        self.theta = theta
        self.ae = ae_model
        self.enc = enc_model
        # self.ohe = ohe
        self.use_kdtree = use_kdtree
        self.batch_size = shape[0]
        self.max_iterations = max_iterations
        self.c_init = c_init
        self.c_steps = c_steps
        self.feature_range = tuple(
            [
                (
                    (np.ones(shape[1:]) * feature_range[_])[None, :]
                    if isinstance(feature_range[_], float)
                    else np.array(feature_range[_])
                )
                for _ in range(2)
            ]
        )
        self.update_num_grad = update_num_grad
        self.eps = eps
        self.clip = clip
        self.input_transform = input_transform
        self.write_dir = write_dir
        self.verbose = verbose

        # define tf variables for original and perturbed instances, and target labels
        self.orig = tf.Variable(np.zeros(shape), dtype=tf.float32, name="orig")
        self.adv = tf.Variable(np.zeros(shape), dtype=tf.float32, name="adv")
        self.adv_s = tf.Variable(np.zeros(shape), dtype=tf.float32, name="adv_s")
        self.target = tf.Variable(
            np.zeros((self.batch_size, self.classes)), dtype=tf.float32, name="target"
        )
        # self.patch = tf.Variable(np.zeros(patch_shape), dtype=tf.float32, name='patch')

        # variable for target class proto
        if self.enc_model:
            self.shape_enc = self.enc.predict(np.zeros(self.shape)).shape  # type: ignore[union-attr]
        else:
            # shape_env may not be equal to shape
            self.shape_enc = self.shape
            # self.shape_enc = self.patch_shape

        self.target_proto = tf.Variable(
            np.zeros(self.shape_enc), dtype=tf.float32, name="target_proto"
        )

        # define tf variable for constant used in FISTA optimization
        self.const = tf.Variable(
            np.zeros(self.batch_size), dtype=tf.float32, name="const"
        )
        self.global_step = tf.Variable(0.0, trainable=False, name="global_step")

        # define placeholders that will be assigned to relevant variables
        self.assign_orig = tf.placeholder(tf.float32, shape, name="assign_orig")
        self.assign_adv = tf.placeholder(tf.float32, shape, name="assign_adv")
        self.assign_adv_s = tf.placeholder(tf.float32, shape, name="assign_adv_s")
        self.assign_target = tf.placeholder(
            tf.float32, (self.batch_size, self.classes), name="assign_target"
        )
        self.assign_const = tf.placeholder(
            tf.float32, [self.batch_size], name="assign_const"
        )
        self.assign_target_proto = tf.placeholder(
            tf.float32, self.shape_enc, name="assign_target_proto"
        )
        # self.assign_patch = tf.placeholder(tf.float32, patch_shape, name='assign_patch')

        # define conditions and values for element-wise shrinkage thresholding
        with tf.name_scope("shrinkage_thresholding") as scope:
            cond = [
                tf.cast(
                    tf.greater(tf.subtract(self.adv_s, self.orig), self.beta),
                    tf.float32,
                ),
                tf.cast(
                    tf.less_equal(
                        tf.abs(tf.subtract(self.adv_s, self.orig)), self.beta
                    ),
                    tf.float32,
                ),
                tf.cast(
                    tf.less(tf.subtract(self.adv_s, self.orig), tf.negative(self.beta)),
                    tf.float32,
                ),
            ]
            upper = tf.minimum(
                tf.subtract(self.adv_s, self.beta),
                tf.cast(feature_range[1], tf.float32),
            )
            lower = tf.maximum(
                tf.add(self.adv_s, self.beta), tf.cast(feature_range[0], tf.float32)
            )
            self.assign_adv = (
                tf.multiply(cond[0], upper)
                + tf.multiply(cond[1], self.orig)
                + tf.multiply(cond[2], lower)
            )

        # perturbation update and vector projection on correct feature range set
        with tf.name_scope("perturbation_y") as scope:
            self.zt = tf.divide(
                self.global_step, self.global_step + tf.cast(3, tf.float32)
            )
            self.assign_adv_s = self.assign_adv + tf.multiply(
                self.zt, self.assign_adv - self.adv
            )
            # map to feature space
            self.assign_adv_s = tf.minimum(
                self.assign_adv_s, tf.cast(feature_range[1], tf.float32)
            )
            self.assign_adv_s = tf.maximum(
                self.assign_adv_s, tf.cast(feature_range[0], tf.float32)
            )

        # assign counterfactual of step k+1 to k
        with tf.name_scope("update_adv") as scope:
            self.adv_updater = tf.assign(self.adv, self.assign_adv)
            self.adv_updater_s = tf.assign(self.adv_s, self.assign_adv_s)

        # from perturbed instance, derive deviation delta
        with tf.name_scope("update_delta") as scope:
            self.delta = self.orig - self.adv
            self.delta_s = self.orig - self.adv_s

        # define L1 and L2 loss terms; L1+L2 is later used as an optimization constraint for FISTA
        ax_sum = list(np.arange(1, len(shape)))
        with tf.name_scope("loss_l1_l2") as scope:
            self.l2 = tf.reduce_sum(tf.square(self.delta), axis=ax_sum)
            self.l2_s = tf.reduce_sum(tf.square(self.delta_s), axis=ax_sum)
            self.l1 = tf.reduce_sum(tf.abs(self.delta), axis=ax_sum)
            self.l1_s = tf.reduce_sum(tf.abs(self.delta_s), axis=ax_sum)
            self.l1_l2 = self.l2 + tf.multiply(self.l1, self.beta)
            self.l1_l2_s = self.l2_s + tf.multiply(self.l1_s, self.beta)

            # sum losses
            self.loss_l1 = tf.reduce_sum(self.l1)
            self.loss_l1_s = tf.reduce_sum(self.l1_s)
            self.loss_l2 = tf.reduce_sum(self.l2)
            self.loss_l2_s = tf.reduce_sum(self.l2_s)

        self.adv_cat = self.adv
        self.adv_cat_s = self.adv_s

        with tf.name_scope("loss_ae") as scope:
            # gamma * AE loss
            if self.ae_model:
                # run autoencoder
                self.adv_ae = self.ae(self.adv_cat)  # type: ignore[misc]
                self.adv_ae_s = self.ae(self.adv_cat_s)  # type: ignore[misc]
                # compute loss
                self.loss_ae = self.gamma * tf.square(tf.norm(self.adv_ae - self.adv))
                self.loss_ae_s = self.gamma * tf.square(
                    tf.norm(self.adv_ae_s - self.adv_s)
                )
            else:  # no auto-encoder available
                self.loss_ae = tf.constant(0.0)
                self.loss_ae_s = tf.constant(0.0)

        with tf.name_scope("loss_attack") as scope:
            if not self.model:
                self.loss_attack = tf.placeholder(tf.float32)
            elif (
                self.c_init == 0.0 and self.c_steps == 1
            ):  # prediction loss term not used
                # make predictions on perturbed instance
                self.pred_proba = self.predict(self.adv_cat)
                self.pred_proba_s = self.predict(self.adv_cat_s)

                self.loss_attack = tf.constant(0.0)
                self.loss_attack_s = tf.constant(0.0)
            else:
                # make predictions on perturbed instance
                self.pred_proba = self.predict(self.adv_cat)
                self.pred_proba_s = self.predict(self.adv_cat_s)

                # probability of target label prediction
                self.target_proba = tf.reduce_sum(self.target * self.pred_proba, 1)
                target_proba_s = tf.reduce_sum(self.target * self.pred_proba_s, 1)

                # max probability of non target label prediction
                self.nontarget_proba_max = tf.reduce_max(
                    (1 - self.target) * self.pred_proba - (self.target * 10000), 1
                )
                nontarget_proba_max_s = tf.reduce_max(
                    (1 - self.target) * self.pred_proba_s - (self.target * 10000), 1
                )

                # loss term f(x,d)
                loss_attack = tf.maximum(
                    0.0, -self.nontarget_proba_max + self.target_proba + self.kappa
                )
                loss_attack_s = tf.maximum(
                    0.0, -nontarget_proba_max_s + target_proba_s + self.kappa
                )

                # c * f(x,d)
                self.loss_attack = tf.reduce_sum(self.const * loss_attack)
                self.loss_attack_s = tf.reduce_sum(self.const * loss_attack_s)

        with tf.name_scope("loss_prototype") as scope:
            # CHANGE: adv is delta, so adv and target_proto are no longer the same shape
            if self.enc_model:
                self.loss_proto = self.theta * tf.square(
                    tf.norm(self.enc(self.adv_cat) - self.target_proto)
                )  # type: ignore[misc]
                self.loss_proto_s = self.theta * tf.square(
                    tf.norm(self.enc(self.adv_cat_s) - self.target_proto)
                )  # type: ignore[misc]
            elif self.use_kdtree:
                self.loss_proto = self.theta * tf.square(
                    tf.norm(self.adv - self.target_proto)
                )
                self.loss_proto_s = self.theta * tf.square(
                    tf.norm(self.adv_s - self.target_proto)
                )
            else:  # no encoder available and no k-d trees used
                self.loss_proto = tf.constant(0.0)
                self.loss_proto_s = tf.constant(0.0)

        with tf.name_scope("loss_combined") as scope:
            # no need for L1 term in loss to optimize when using FISTA
            if self.model:
                self.loss_opt = (
                    self.loss_attack_s
                    + self.loss_l2_s
                    + self.loss_ae_s
                    + self.loss_proto_s
                )
            else:  # separate numerical computation of loss attack gradient
                self.loss_opt = self.loss_l2_s + self.loss_ae_s + self.loss_proto_s

            # add L1 term to overall loss; this is not the loss that will be directly optimized
            self.loss_total = (
                self.loss_attack
                + self.loss_l2
                + self.loss_ae
                + tf.multiply(self.beta, self.loss_l1)
                + self.loss_proto
            )

        with tf.name_scope("training") as scope:
            self.learning_rate = tf.train.polynomial_decay(
                learning_rate_init, self.global_step, self.max_iterations, 0, power=0.5
            )
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            start_vars = set(x.name for x in tf.global_variables())

            # first compute, then apply grads
            self.compute_grads = optimizer.compute_gradients(
                self.loss_opt, var_list=[self.adv_s]
            )
            self.grad_ph = tf.placeholder(tf.float32, name="grad_adv_s")
            var = [
                tvar
                for tvar in tf.trainable_variables()
                if tvar.name.startswith("adv_s")
            ][
                -1
            ]  # get the last in
            # case explainer is re-initialized and a new graph is created
            grad_and_var = [(self.grad_ph, var)]
            self.apply_grads = optimizer.apply_gradients(
                grad_and_var, global_step=self.global_step
            )
            end_vars = tf.global_variables()
            new_vars = [x for x in end_vars if x.name not in start_vars]

        # variables to initialize
        self.setup = []  # type: list
        self.setup.append(self.orig.assign(self.assign_orig))
        self.setup.append(self.target.assign(self.assign_target))
        self.setup.append(self.const.assign(self.assign_const))
        self.setup.append(self.adv.assign(self.assign_adv))
        self.setup.append(self.adv_s.assign(self.assign_adv_s))
        self.setup.append(self.target_proto.assign(self.assign_target_proto))
        # self.setup.append(self.patch.assign(self.assign_patch))
        # if self.is_cat:
        #     self.setup.append(self.map_var.assign(self.assign_map))

        self.init = tf.variables_initializer(
            var_list=[self.global_step] + [self.adv_s] + [self.adv] + new_vars
        )

        if self.write_dir is not None:
            self.writer = tf.summary.FileWriter(write_dir, tf.get_default_graph())
            self.writer.add_graph(tf.get_default_graph())
        else:
            self.writer = None

    def fit(
        self,
        train_data: np.ndarray = np.array([]),
        preds: np.ndarray = np.array([]),
        trustscore_kwargs: Optional[dict] = None,
    ) -> "Counterfactual":
        """
        Get prototypes for each class using the encoder or k-d trees.
        The prototypes are used for the encoder loss term or to calculate the optional trust scores.

        Parameters
        ----------
        train_data
            Representative sample from the training data.
        trustscore_kwargs
            Optional arguments to initialize the trust scores method.
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
            ts = load_object(self.trustscore)
            self.kdtrees = ts.kdtrees
            self.X_by_class = ts.X_kdtree
        return self

    def loss_fn(self, pred_proba: np.ndarray, Y: np.ndarray) -> np.ndarray:
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
        target_proba = np.sum(pred_proba * Y)
        # max probability of non target label prediction
        nontarget_proba_max = np.max((1 - Y) * pred_proba - 10000 * Y)
        # loss term f(x,d)
        loss = np.maximum(0.0, -nontarget_proba_max + target_proba + self.kappa)
        # c * f(x,d)
        loss_attack = np.sum(self.const.eval(session=self.sess) * loss)
        return loss_attack

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
        X_pred = torch.tensor(X)
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
            logger.warning(
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
        verbose: bool = False,
        print_every: int = 100,
        log_every: int = 100,
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
        verbose
            Print intermediate results of optimization if ``True``.
        print_every
            Print frequency if verbose is ``True``.
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
            if not isinstance(x, (float, int, np.int64)):
                x = np.copy(x)
                x[y] += self.kappa  # type: ignore
                x = np.argmax(x)  # type: ignore
            return x != y

        # define target classes for prototype if not specified yet
        if target_class is None:
            target_class = list(range(self.classes))
            target_class.remove(np.argmax(Y, axis=1))
            if verbose:
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
                # Xpatch = self.input_transform(X_num).eval(session=self.sess)
                # Xpatch = Xpatch.reshape(Xpatch.shape[0], -1)
                dist_c, idx_c = self.kdtrees[c].query(X_num, k=k)
                dist_proto[c] = dist_c[0][-1]
                self.class_proto[c] = self.X_by_class[c][idx_c[0][-1]].reshape(1, -1)

        if self.enc_or_kdtree:
            self.id_proto = min(dist_proto, key=dist_proto.get)  # type: ignore[arg-type]
            # proto_val = self.class_proto[self.id_proto].reshape(self.patch_shape)
            proto_val = self.class_proto[self.id_proto]
            if verbose:
                print("Prototype class: {}".format(self.id_proto))
        else:  # no prototype loss term used
            proto_val = np.zeros(self.shape_enc)

        # set shape for perturbed instance and gradients
        pert_shape = self.shape

        # set the lower and upper bounds for the constant 'c' to scale the attack loss term
        # these bounds are updated for each c_step iteration
        const_lb = np.zeros(self.batch_size)
        const = np.ones(self.batch_size) * self.c_init
        const_ub = np.ones(self.batch_size) * 1e10

        # init values for the best attack instances for each instance in the batch
        overall_best_dist = [1e10] * self.batch_size
        overall_best_attack = [np.zeros(self.shape[1:])] * self.batch_size
        overall_best_grad = (np.zeros(self.shape), np.zeros(self.shape))

        # keep track of counterfactual evolution
        self.cf_global = {i: [] for i in range(self.c_steps)}  # type: dict

        # iterate over nb of updates for 'c'
        for _ in range(self.c_steps):

            # init variables
            self.sess.run(self.init)

            # reset current best distances and scores
            current_best_dist = [1e10] * self.batch_size
            current_best_proba = [-1] * self.batch_size

            # assign variables for the current iteration
            feed_dict = {
                self.assign_orig: X_num,
                self.assign_target: Y,
                self.assign_const: const,
                self.assign_adv: X_num,
                self.assign_adv_s: X_num,
                self.assign_target_proto: proto_val,
                # self.assign_patch: patch[0],
            }
            self.sess.run(self.setup, feed_dict=feed_dict)

            X_der_batch, X_der_batch_s = [], []  # type: Any, Any

            for i in range(self.max_iterations):

                # numerical gradients
                grads_num = np.zeros(pert_shape)
                grads_num_s = np.zeros(pert_shape)

                # check if numerical gradient computation is needed
                if not self.model and (self.c_init != 0.0 or self.c_steps > 1):
                    X_der = self.adv.eval(session=self.sess)
                    X_der_s = self.adv_s.eval(session=self.sess)
                    X_der_batch.append(X_der)
                    X_der_batch_s.append(X_der_s)

                    if (
                        i % self.update_num_grad == 0 and i > 0
                    ):  # compute numerical gradients
                        c = self.const.eval(session=self.sess)
                        X_der_batch = np.concatenate(X_der_batch)
                        X_der_batch_s = np.concatenate(X_der_batch_s)
                        grads_num = (
                            self.get_gradients(
                                X_der_batch, Y, grads_shape=pert_shape[1:]
                            )
                            * c
                        )
                        grads_num_s = (
                            self.get_gradients(
                                X_der_batch_s, Y, grads_shape=pert_shape[1:]
                            )
                            * c
                        )
                        # clip gradients
                        grads_num = np.clip(grads_num, self.clip[0], self.clip[1])  # type: ignore
                        grads_num_s = np.clip(grads_num_s, self.clip[0], self.clip[1])  # type: ignore
                        X_der_batch, X_der_batch_s = [], []

                # compute and clip gradients defined in graph
                grads_vars_graph = self.sess.run(self.compute_grads)
                grads_graph = [g for g, _ in grads_vars_graph][0]
                grads_graph = np.clip(grads_graph, self.clip[0], self.clip[1])

                # apply gradients
                grads = grads_graph + grads_num_s
                self.sess.run(self.apply_grads, feed_dict={self.grad_ph: grads})

                # update adv and adv_s with perturbed instances
                self.sess.run(
                    [self.adv_updater, self.adv_updater_s, self.delta, self.delta_s]
                )

                # compute overall and attack loss, L1+L2 loss, prediction probabilities
                # on perturbed instances and new adv
                # L1+L2 and prediction probabilities used to see if adv is better than the current best adv under FISTA
                if self.model:
                    loss_tot, loss_attack, loss_l1_l2, pred_proba, adv = self.sess.run(
                        [
                            self.loss_total,
                            self.loss_attack,
                            self.l1_l2,
                            self.pred_proba,
                            self.adv,
                        ]
                    )
                else:
                    X_der = self.adv.eval(
                        session=self.sess
                    )  # get updated perturbed instances
                    # convert X_der to torch tensor
                    X_der = torch.tensor(X_der, dtype=torch.float32)
                    pred_proba = self.predict(X_der).detach().numpy()

                    # compute attack, total and L1+L2 losses as well as new perturbed instance
                    loss_attack = self.loss_fn(pred_proba, Y)
                    feed_dict = {self.loss_attack: loss_attack}
                    loss_tot, loss_l1_l2, adv = self.sess.run(
                        [self.loss_total, self.l1_l2, self.adv], feed_dict=feed_dict
                    )

                if i % log_every == 0 or i % print_every == 0:
                    loss_l2, loss_l1, loss_ae, loss_proto = self.sess.run(
                        [self.loss_l2, self.loss_l1, self.loss_ae, self.loss_proto]
                    )
                    target_proba = np.sum(pred_proba * Y)
                    nontarget_proba_max = np.max((1 - Y) * pred_proba)
                    loss_opt = loss_l1_l2 + loss_attack + loss_ae + loss_proto

                if i % log_every == 0 and self.writer is not None:
                    lr, zt, gs = self.sess.run(
                        [self.learning_rate, self.zt, self.global_step]
                    )

                    # add values and images to tensorboard
                    summary = tf.Summary()
                    summary.value.add(tag="loss/Optimized", simple_value=loss_opt)
                    summary.value.add(tag="loss/Total", simple_value=loss_tot)
                    summary.value.add(tag="loss/L1", simple_value=loss_l1)
                    summary.value.add(tag="loss/L2", simple_value=loss_l2)
                    summary.value.add(tag="loss/AutoEncoder", simple_value=loss_ae)
                    summary.value.add(
                        tag="loss/ClassPrototype", simple_value=loss_proto
                    )
                    summary.value.add(tag="loss/PredScale", simple_value=const[0])
                    summary.value.add(tag="loss/PredLoss", simple_value=loss_attack)
                    summary.value.add(tag="training/lr", simple_value=lr)
                    summary.value.add(tag="training/z", simple_value=zt)
                    summary.value.add(tag="training/GlobalStep", simple_value=gs)
                    self.writer.add_summary(summary)
                    self.writer.flush()

                if verbose and i % print_every == 0:
                    print("\nIteration: {}; Const: {}".format(i, const[0]))
                    print(
                        "Loss total: {:.3f}, loss attack: {:.3f}".format(
                            loss_tot, loss_attack
                        )
                    )
                    print(
                        "L2: {:.3f}, L1: {:.3f}, loss AE: {:.3f}".format(
                            loss_l2, loss_l1, loss_ae
                        )
                    )
                    print("Loss proto: {:.3f}".format(loss_proto))
                    print(
                        "Target proba: {:.2f}, max non target proba: {:.2f}".format(
                            target_proba, nontarget_proba_max
                        )
                    )
                    print(
                        "Gradient graph min/max: {:.3f}/{:.3f}".format(
                            grads_graph.min(), grads_graph.max()
                        )
                    )
                    print(
                        "Gradient graph mean/abs mean: {:.3f}/{:.3f}".format(
                            np.mean(grads_graph), np.mean(np.abs(grads_graph))
                        )
                    )  # type: ignore
                    if not self.model:
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
                    zip(loss_l1_l2, pred_proba, adv)
                ):
                    Y_class = np.argmax(Y[batch_idx])
                    adv_class = np.argmax(proba)
                    adv_idx = np.expand_dims(adv_idx, axis=0)

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
                        if verbose:
                            print("\nNew best counterfactual found!")
                        overall_best_dist[batch_idx] = dist
                        overall_best_attack[batch_idx] = adv_idx
                        overall_best_grad = (grads_graph, grads_num)
                        self.best_attack = True
                        self.cf_global[_].append(adv_idx)

            # adjust the 'c' constant for the first loss term
            for batch_idx in range(self.batch_size):
                if (
                    compare(current_best_proba[batch_idx], np.argmax(Y[batch_idx]))  # type: ignore
                    and current_best_proba[batch_idx] != -1
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
        best_attack = np.concatenate(overall_best_attack, axis=0)
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
        log_every: int = 100,
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
        verbose
            Print intermediate results of optimization if ``True``.
        print_every
            Print frequency if verbose is ``True``.
        log_every
            `tensorboard` log frequency if write directory is specified

        Returns
        -------
        explanation
            `Explanation` object containing the counterfactual with additional metadata as attributes.
        """
        # get params for storage in meta
        params = locals()
        remove = ["self", "X", "Y"]
        for key in remove:
            params.pop(key)

        if X.shape[0] != 1:
            logger.warning(
                "Currently only single instance explanations supported (first dim = 1), "
                "but first dim = %s",
                X.shape[0],
            )

        # output explanation dictionary
        data = copy.deepcopy(DEFAULT_DATA)

        # if Y is None:
        if self.model:
            Y_proba = self.predict.predict(X)  # type: ignore
        else:
            Y_proba = self.predict(X)
        Y_ohe = np.zeros(Y_proba.shape)
        Y_class = np.argmax(Y_proba.detach(), axis=1)
        Y_ohe[np.arange(Y_proba.shape[0]), Y_class] = 1
        Y = Y_ohe.copy()
        data["orig_proba"] = Y_proba
        # else:  # provided one-hot-encoding of prediction on X
        # data["orig_proba"] = None
        data["orig_class"] = np.argmax(Y, axis=1)[0]

        # find best counterfactual
        self.best_attack = False
        best_attack, grads = self.attack(
            X,
            Y=Y,
            target_class=target_class,
            k=k,
            k_type=k_type,
            verbose=self.verbose,
            threshold=threshold,
            print_every=print_every,
            log_every=log_every,
        )

        if self.enc_or_kdtree:
            data["id_proto"] = self.id_proto

        # add to explanation dict
        if not self.best_attack:
            logger.warning("No counterfactual found!")

            # create explanation object
            explanation = Explanation(meta=copy.deepcopy(self.meta), data=data)
            return explanation

        data["all"] = self.cf_global
        data["cf"] = {}
        data["cf"]["X"] = best_attack
        Y_pert = self.predict(torch.tensor(best_attack)).detach().numpy()
        data["cf"]["class"] = np.argmax(Y_pert, axis=1)[0]
        data["cf"]["proba"] = Y_pert
        data["cf"]["grads_graph"], data["cf"]["grads_num"] = grads[0], grads[1]

        # create explanation object
        explanation = Explanation(meta=copy.deepcopy(self.meta), data=data)

        return explanation


def save_object(obj, filename):
    with open(filename, "wb") as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, -1)


def load_object(filename):
    with open(filename, "rb") as outp:
        return pickle.load(outp)
