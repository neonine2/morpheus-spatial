import torch
import torch.nn as nn
import torch.optim as optim


class Counterfactual:
    def __init__(
        self,
        model,
        shape,
        kappa=0.0,
        beta=0.1,
        gamma=0.0,
        theta=0.0,
        ae_model=None,
        enc_model=None,
        feature_range=(0, 1),
        c_init=1.0,
        c_steps=5,
        eps=(1e-3, 1e-3),
        clip=(-1000.0, 1000.0),
        update_num_grad=1,
        max_iterations=500,
        lr=1e-2,
    ):
        """
        Initialize the Counterfactual class.

        Parameters
        ----------
        model : torch.nn.Module
            The PyTorch model to generate counterfactuals for.
        shape : tuple
            The shape of the input data (excluding batch dimension).
        kappa : float, optional
            Confidence parameter for the attack loss term (default: 0).
        beta : float, optional
            Regularization constant for L1 loss term (default: 0.1).
        gamma : float, optional
            Regularization constant for optional auto-encoder loss term (default: 0).
        theta : float, optional
            Constant for the prototype search loss term (default: 0).
        ae_model : torch.nn.Module, optional
            Optional auto-encoder model used for loss regularization (default: None).
        enc_model : torch.nn.Module, optional
            Optional encoder model used to guide instance perturbations (default: None).
        feature_range : tuple, optional
            Tuple with min and max ranges for features (default: (0, 1)).
        c_init : float, optional
            Initial value to scale the attack loss term (default: 1.0).
        c_steps : int, optional
            Number of iterations to adjust the constant scaling the attack loss term (default: 5).
        eps : tuple, optional
            Tuple with perturbation size used for numerical gradients (default: (1e-3, 1e-3)).
        clip : tuple, optional
            Tuple with min and max clip ranges for gradients (default: (-1000., 1000.)).
        update_num_grad : int, optional
            Number of iterations after which to update numerical gradients (default: 1).
        max_iterations : int, optional
            Maximum number of iterations for counterfactual search (default: 500).
        lr : float, optional
            Learning rate for the optimizer (default: 1e-2).
        """
        self.model = model
        self.shape = shape
        self.kappa = kappa
        self.beta = beta
        self.gamma = gamma
        self.theta = theta
        self.ae_model = ae_model
        self.enc_model = enc_model
        self.feature_range = feature_range
        self.c_init = c_init
        self.c_steps = c_steps
        self.eps = eps
        self.clip = clip
        self.update_num_grad = update_num_grad
        self.max_iterations = max_iterations
        self.lr = lr

        # Initialize variables
        self.const = torch.tensor(c_init, requires_grad=False)
        self.global_step = 0
        self.adv = None
        self.adv_s = None
        self.best_dist = None
        self.best_score = None
        self.best_attack = None

    def generate_counterfactual(self, x, y, target_class=None, verbose=False):
        """
        Generate a counterfactual for the given input instance.

        Parameters
        ----------
        x : torch.Tensor
            The input instance to generate a counterfactual for.
        y : torch.Tensor
            The target label for the input instance.
        target_class : int, optional
            The target class for the counterfactual (default: None).
        verbose : bool, optional
            Whether to print progress during optimization (default: False).

        Returns
        -------
        torch.Tensor
            The generated counterfactual instance.
        """
        # Initialize the counterfactual search
        self.adv = x.clone().detach().requires_grad_(True)
        self.adv_s = x.clone().detach().requires_grad_(True)
        self.best_dist = float("inf")
        self.best_score = -float("inf")
        self.best_attack = x.clone().detach()

        # Define the target class for the counterfactual
        if target_class is None:
            target_class = (y.argmax(dim=1) + 1) % y.shape[1]

        # Optimize the counterfactual
        optimizer = optim.Adam([self.adv], lr=self.lr)
        for i in range(self.max_iterations):
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass through the model
            logits = self.model(self.adv)
            pred_proba = nn.functional.softmax(logits, dim=1)
            pred_class = pred_proba.argmax(dim=1)

            # Compute the loss terms
            loss_attack = self.attack_loss(pred_proba, y, target_class)
            loss_l1 = self.l1_loss(self.adv, x)
            loss_l2 = self.l2_loss(self.adv, x)
            loss_ae = self.ae_loss(self.adv, x) if self.ae_model else 0.0
            loss_proto = (
                self.proto_loss(self.adv, target_class) if self.enc_model else 0.0
            )

            loss = (
                loss_attack
                + self.beta * loss_l1
                + loss_l2
                + self.gamma * loss_ae
                + self.theta * loss_proto
            )

            # Backward pass and optimization step
            loss.backward()
            optimizer.step()

            # Clip the adversarial example to stay within feature range
            self.adv.data = self.adv.data.clamp(*self.feature_range)

            # Update the best counterfactual found so far
            if pred_class == target_class and loss_l1 + loss_l2 < self.best_dist:
                self.best_dist = loss_l1 + loss_l2
                self.best_score = loss_attack
                self.best_attack = self.adv.clone().detach()

            # Print progress if verbose
            if verbose and i % 100 == 0:
                print(
                    f"Iteration {i}: Loss = {loss:.4f}, Best dist = {self.best_dist:.4f}"
                )

            # Update the constant for the attack loss term
            if i % self.update_num_grad == 0:
                self.update_const()

        return self.best_attack

    def attack_loss(self, pred_proba, y, target_class):
        """
        Compute the attack loss term.
        """
        target_proba = pred_proba[:, target_class]
        other_proba = torch.cat(
            (pred_proba[:, :target_class], pred_proba[:, target_class + 1 :]), dim=1
        )
        other_proba_max = other_proba.max(dim=1)[0]
        loss = torch.max(torch.tensor(0.0), other_proba_max - target_proba + self.kappa)
        return loss.mean()

    def l1_loss(self, adv, x):
        """
        Compute the L1 loss term.
        """
        return torch.abs(adv - x).sum(dim=1).mean()

    def l2_loss(self, adv, x):
        """
        Compute the L2 loss term.
        """
        return torch.pow(adv - x, 2).sum(dim=1).mean()

    def ae_loss(self, adv, x):
        """
        Compute the auto-encoder loss term.
        """
        rec_adv = self.ae_model(adv)
        rec_x = self.ae_model(x)
        return torch.pow(rec_adv - rec_x, 2).sum(dim=1).mean()

    def proto_loss(self, adv, target_class):
        """
        Compute the prototype search loss term.
        """
        enc_adv = self.enc_model(adv)
        proto = self.get_prototype(target_class)
        return torch.pow(enc_adv - proto, 2).sum(dim=1).mean()

    def get_prototype(self, target_class):
        """
        Get the prototype for the target class.
        """
        # Implement the prototype selection logic based on your specific needs
        # For example, you can use the average encoding of instances belonging to the target class
        # You can also use k-d trees to find the nearest prototype if an encoder is not available
        raise NotImplementedError("Prototype selection logic not implemented")

    def update_const(self):
        """
        Update the constant for the attack loss term.
        """
        # Implement the logic to update the constant based on the counterfactual search progress
        # For example, you can increase or decrease the constant based on the loss values
        raise NotImplementedError("Constant update logic not implemented")
