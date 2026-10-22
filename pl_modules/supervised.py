from pl_modules.base import BaseModel
from typing import Dict, Optional
import torch.nn as nn
import sys
import math
import torch
from torchmetrics.classification import MultilabelAccuracy, F1Score
from torchmetrics.regression import R2Score, PearsonCorrCoef, MeanSquaredError
from losses.supcon import SupConLoss


class SupervisedClassifier(BaseModel):
    """
        Supervised contrastive classifier model on top of a pre-trained feature extractor.
    """

    def __init__(self, ckpt_path: str,
                 num_classes: int,
                 optim_kwargs: Dict,
                 model: str = "supcon",
                 supcon_head: bool = True):
        """
        :param supcon_head: if True, SupCon acts on a projection head trained from scratch, with
            the architecture of the pre-training head, as in Khosla et al.; the probes still read
            the encoder output. False applies it to the encoder output itself, as CoMM's code does.
        """

        super().__init__(optim_kwargs)
        assert model in {"supcon", "bce_with_logits", "mse"}
        self.ckpt_path = ckpt_path
        self.num_classes = num_classes
        self.base_encoder, embed_dim = self._parse_ckpt(self.ckpt_path)
        self.model = model
        self.metrics = dict()
        self.head = None
        if self.model == "supcon":
            self.loss = SupConLoss()
            if supcon_head:
                # The pre-trained head is taken out of the encoder and re-initialised, so the
                # fine-tuned model keeps the parameter count of the pre-trained one.
                self.head = self.base_encoder.head
                del self.base_encoder.head
                for m in self.head.modules():
                    if m is not self.head and hasattr(m, "reset_parameters"):
                        m.reset_parameters()
        elif self.model == "bce_with_logits":
            self.loss = nn.BCEWithLogitsLoss()
            self.linear = nn.Linear(embed_dim, num_classes)
            self.metrics.update({
                "acc1": MultilabelAccuracy(num_labels=num_classes,
                                           average="macro").to("cuda"),
                "f1_mean": F1Score(task="multilabel", num_labels=num_classes,
                                    average="macro").to("cuda"),
                "f1_weighted": F1Score(task="multilabel", num_labels=num_classes,
                                             average="weighted").to("cuda")
            })
        elif self.model == "mse":
            self.loss = nn.MSELoss()
            self.linear = nn.Linear(embed_dim, num_classes)
            self.metrics.update({
                "pearson_r": lambda y_pred, y: PearsonCorrCoef(num_outputs=num_classes).to("cuda")(y_pred, y).mean(),
                "mse": lambda y_pred, y: MeanSquaredError(num_outputs=num_classes).to("cuda")(y_pred, y).mean(),
                "r2": R2Score(num_outputs=num_classes, multioutput="uniform_average").to("cuda")
            })
        else:
            raise NotImplementedError()

    def _parse_ckpt(self, pth: str):
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        ckpt = torch.load(pth, map_location="cpu", weights_only=False)
        # As a config, so that interpolations such as `${embed_dim}` in the saved
        # encoders resolve against the run's own root.
        hparams = OmegaConf.create(ckpt["hyper_parameters"])
        dataset = hparams.data.data_module.get("dataset", hparams.data.name)
        model_name = hparams.model.name
        kwargs = dict()
        if model_name in ("CoMM", "WoMM"):
            encoders = instantiate(hparams[dataset]["encoders"])  # encoders specific to each dataset
            adapters = instantiate(hparams[dataset]["adapters"])  # adapters also specific
            kwargs["encoder"] = {
                "encoders": encoders,
                "input_adapters": adapters}
            model = instantiate(hparams.model.model, optim_kwargs=hparams.optim, **kwargs)
            embed_dim = int(hparams.model.model.encoder.embed_dim)
            if hasattr(model, "on_load_checkpoint"):
                model.on_load_checkpoint(ckpt)  # WoMM restores the keys of frozen encoders
            model.load_state_dict(ckpt["state_dict"], strict=True)
        else:
            raise NotImplementedError()
        return model, embed_dim

    def forward(self, X):
        if self.model == "supcon":
            X1, X2 = X
            embed1 = self.base_encoder.encoder(X1)
            embed2 = self.base_encoder.encoder(X2)
            if self.head is not None:
                embed1, embed2 = self.head(embed1), self.head(embed2)
            return embed1, embed2
        elif self.model in ["bce_with_logits", "mse"]:
            embed = self.base_encoder.encoder(X)
            out = self.linear(embed)
            return out
        raise NotImplementedError()

    def training_step(self, batch, batch_idx):
        if self.model == "supcon":
            X1, X2, y = batch
            Z1, Z2 = self.forward((X1, X2))
            Z = torch.stack((Z1, Z2), dim=1)
            loss = self.loss(Z, y)
        else:
            X, y = batch
            y_pred = self.forward(X)
            loss = self.loss(y_pred, y.float())
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)
        self.log_dict({"loss": loss}, on_epoch=True,
                      sync_dist=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        metrics = dict()
        if self.model == "supcon":
            X1, X2, y = batch
            Z1, Z2 = self.forward((X1, X2))
            Z = torch.stack((Z1, Z2), dim=1)
            val_loss = self.loss(Z, y)
        else:
            X, y = batch
            y_pred = self.forward(X)
            val_loss = self.loss(y_pred, y.float())
            for metric_name, metric in self.metrics.items():
                metrics[metric_name] = float(metric(y_pred, y))
        self.log_dict({"val_loss": val_loss, **metrics},
                      on_epoch=True, sync_dist=True, prog_bar=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        metrics = dict()
        if self.model == "supcon":
            X1, X2, y = batch
            Z1, Z2 = self.forward((X1, X2))
            Z = torch.stack((Z1, Z2), dim=1)
            test_loss = self.loss(Z, y)
        else:
            X, y = batch
            y_pred = self.forward(X)
            test_loss = self.loss(y_pred, y.float())
            for metric_name, metric in self.metrics.items():
                metrics[metric_name] = float(metric(y_pred, y))
        self.log_dict({"test_loss": test_loss, **metrics},
                      on_epoch=True, sync_dist=True, prog_bar=True)
        return test_loss

    def extract_features(self, loader: torch.utils.data.DataLoader, **kwargs):
        return self.base_encoder.extract_features(loader, **kwargs)