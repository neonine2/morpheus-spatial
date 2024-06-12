import lightning as light
import torch
import torchmetrics.functional.classification as tf_classifier
from torch import nn
from torch.nn import functional


class PatchClassifier(light.LightningModule):
    def __init__(
        self,
        in_channels,
        img_size,
        arch="unet",
        num_target_classes=2,
        optimizer="adam",
        optimizer_params=None,
    ):
        super().__init__()
        self.predictor = None
        if optimizer_params is None:
            optimizer_params = {"lr": 1e-3}
        self.classes = num_target_classes
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.arch = arch.lower()

        # save hyperparameters
        self.save_hyperparameters()

        # build model
        self.build_model(in_channels, img_size)

    def build_model(self, in_channels, img_size):
        """
        Selects and builds the chosen model architecture.
        """
        if self.arch == "unet":
            backbone = torch.hub.load(
                "mateuszbuda/brain-segmentation-pytorch",
                "unet",
                in_channels=in_channels,
                out_channels=1,
                init_features=in_channels,
                verbose=False,
            )
            classifier = torch.nn.Sequential()
            classifier.add_module("flatten", nn.Flatten())
            classifier.add_module(
                "fc", nn.Linear(img_size[0] * img_size[1], self.classes)
            )
            classifier.add_module("act", nn.Softmax(dim=1))
            self.predictor = nn.Sequential(*[backbone, classifier])
        elif self.arch == "mlp":
            self.predictor = nn.Sequential(
                nn.Linear(in_channels, 30),
                nn.ReLU(),
                nn.Linear(30, 10),
                nn.ReLU(),
                nn.Linear(10, self.classes),
                nn.Softmax(dim=1),
            )
        elif self.arch == "lr":
            self.predictor = nn.Sequential(nn.Linear(in_channels, 1), nn.Sigmoid())
        else:
            raise ValueError(f"Invalid model architecture: {self.model_arch}")

    def forward(self, x):
        self.predictor.eval()
        return self.predictor(x)

    def configure_optimizers(self):
        if self.optimizer == "adam":
            return torch.optim.Adam(self.parameters(), **self.optimizer_params)

    def execute_and_get_metric(self, batch, mode):
        x, target = batch
        target = functional.one_hot(target, num_classes=self.classes).float()
        predictor = self.predictor(x)
        return self.log_metrics(mode, predictor, target)

    def training_step(self, train_batch, batch_idx):
        metric_dict = self.execute_and_get_metric(train_batch, "train")
        self.log_dict(metric_dict, on_step=False, on_epoch=True, prog_bar=False)
        return metric_dict["train_bce"]

    def validation_step(self, val_batch, batch_idx):
        metric_dict = self.execute_and_get_metric(val_batch, "val")
        self.log_dict(metric_dict, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, test_batch, batch_idx):
        metric_dict = self.execute_and_get_metric(test_batch, "test")
        self.log_dict(metric_dict, prog_bar=True)

    @staticmethod
    def log_metrics(mode, preds, target):
        # classification metrics
        if preds.shape[1] == 1:
            # Create a new column that is 1 minus the first column
            new_col = 1 - preds[:, 0]
            # Append the new column to the original matrix
            preds = torch.column_stack((preds, new_col))
        bce = functional.binary_cross_entropy_with_logits(preds, target)

        auroc = tf_classifier.binary_auroc(
            preds[:, 1], target[:, 1].long()
        )  # keep as probabilities
        preds = torch.argmax(preds, dim=1).float()
        target = torch.argmax(target, dim=1).float()
        test_acc = tf_classifier.binary_accuracy(preds, target)
        bmc = tf_classifier.binary_matthews_corrcoef(preds, target).float()
        f1 = tf_classifier.binary_f1_score(preds, target)
        precision = tf_classifier.binary_precision(preds, target)
        recall = tf_classifier.binary_recall(preds, target)
        metric_dict = {
            mode + "_bce": bce,
            mode + "_precision": precision,
            mode + "_recall": recall,
            mode + "_bmc": bmc,
            mode + "_auroc": auroc,
            mode + "_f1": f1,
            mode + "_acc": test_acc,
        }
        return metric_dict


def load_model(model_path: str, eval: bool = True, **kwargs):
    """
    Load the trained model.

    Args:
        model_path (str): Path to the model checkpoint.

    Returns:
        torch.nn.Module: Loaded model.
    """
    model = PatchClassifier.load_from_checkpoint(model_path, **kwargs)
    if eval:
        model.eval()
    return model


def get_prediction(model, data_loader):
    m = nn.Softmax(dim=1)
    preds = []
    labels = []
    for x, y in iter(data_loader):
        pred = model(x)
        if pred.shape[1] == 1:
            # Create a new column that is 1 minus the first column
            new_col = 1 - pred[:, 0]
            # Append the new column to the original matrix
            pred = torch.column_stack((pred, new_col))
        preds.append(m(pred)[:, 1])
        labels.append(y)
    preds = torch.cat(preds, dim=0)
    labels = torch.cat(labels, dim=0)
    return preds, labels
