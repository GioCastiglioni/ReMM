from omegaconf import DictConfig
import hydra
from hydra.utils import instantiate
import os
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from utils import setup_results_dir, build_run_identity, WANDB_PROJECT


# Datasets whose downstream target is continuous: they need the regression probe.
REGRESSION_DATASETS = ("visionandtouch",)


def modality_masks(n_modalities: int):
    """Probe masks: the joint representation plus one modality at a time.

    Reading a task from a single modality is the falsifiability check of the PID
    reading: a unique attribute must be legible from its own modality only, and a
    synergistic one from neither alone. The joint mask keeps the name `both` for
    two modalities so the metric keys match the bimodal datasets already run.
    """
    joint = "both" if n_modalities == 2 else "all"
    masks = {joint: [True] * n_modalities}
    for i in range(n_modalities):
        masks[f"mod{i + 1}"] = [j == i for j in range(n_modalities)]
    return masks


@hydra.main(version_base=None, config_name="train_multibench", config_path="./configs")
def main(cfg: DictConfig):
    """Training/test of Multi-Modal models on MultiBench datasets.

    Models currently implemented are:
        - CoMM [ours!]
        - WoMM [ours!]
        - CrossSelf
        - CLIP
        - SupervisedClassifier (from pretrained model)
    """

    # fix the seed for repro
    pl.seed_everything(cfg.seed, workers=True)

    # create model + save hyper-parameters
    dataset = cfg.data.data_module.dataset # Which MultiBench dataset to load

    # Appendix B gives a different embedding space per dataset, so each block declares
    # its own. `model/*.yaml` reads `embed_dim` from the config root, which holds only a
    # fallback, so copy it up before the model is built.
    if "embed_dim" in cfg[dataset]:
        cfg.embed_dim = cfg[dataset].embed_dim
    # Appendix B.4 also gives a different learning rate per dataset.
    if "optim" in cfg[dataset]:
        cfg.optim = cfg[dataset].optim
    print(f"[main] {dataset}: embed_dim={cfg.embed_dim} "
          f"lr={cfg.optim.lr} wd={cfg.optim.weight_decay}")

    kwargs = dict()
    if cfg.model.name in ("CoMM", "WoMM", "MMSD"):
        encoders = instantiate(cfg[dataset]["encoders"]) # encoders specific to each dataset
        adapters = instantiate(cfg[dataset]["adapters"]) # adapters also specific
        kwargs["encoder"] = {
            "encoders": encoders,
            "input_adapters": adapters}
    elif cfg.model.name == "CLIP":
        encoders = instantiate(cfg[dataset]["encoders"]) # encoders specific to each dataset
        kwargs["visual"], kwargs["language"] = encoders[0], encoders[1]
        kwargs["image_projection"] = instantiate(cfg[dataset].clip_projection1)
        kwargs["text_projection"] = instantiate(cfg[dataset].clip_projection2)
    elif cfg.model.name == "CrossSelf":
        encoders = instantiate(cfg[dataset]["encoders"])
        kwargs["enc1"] = encoders[0]
        kwargs["enc2"] = encoders[1]
        kwargs["head1"] = instantiate(cfg[dataset].projection_head1)
        kwargs["head2"] = instantiate(cfg[dataset].projection_head2)

    model = instantiate(cfg.model.model, optim_kwargs=cfg.optim, **kwargs)
    model.save_hyperparameters(cfg)

    # Data loading code
    modalities = list(cfg[dataset]["modalities"])
    data_module = instantiate(cfg.data.data_module,
                              model=cfg.model.name,
                              modalities=modalities,
                              task=cfg[dataset]["task"],
                              **cfg[dataset]["kwargs"])

    downstream_data_module = instantiate(cfg.data.data_module,
                                         model="Sup",
                                         modalities=modalities,
                                         task=cfg[dataset]["task"])

    # One probe per mask. `always_prefix` is required even with a single name:
    # without it every callback would log the bare key `acc1` and overwrite the
    # others.
    probe_cfg = cfg.linear_probing_reg if dataset in REGRESSION_DATASETS else cfg.linear_probing
    masks = modality_masks(len(modalities))
    finetune = cfg.model.name == "SupervisedClassifier"
    if finetune:
        # Fine-tuning reports the joint representation, probed at every epoch for early
        # stopping, so the unimodal probes would only triple that cost.
        masks = dict(list(masks.items())[:1])
    callbacks = [instantiate(probe_cfg,
                             downstream_data_modules=[downstream_data_module],
                             names=[f"{dataset}_{m}"],
                             mask_modalities=[mask],
                             always_prefix=True,
                             _convert_="all")
                 for m, mask in masks.items()]

    # Resuming: `ckpt_path` continues training from a finished run and `wandb_id`
    # keeps the curves in that run instead of opening a second one.
    resume_ckpt = getattr(cfg, "ckpt_path", None) if cfg.mode == "train" else None
    wandb_id = getattr(cfg, "wandb_id", None)
    wandb_resume = {"id": wandb_id, "resume": "allow"} if wandb_id else {}

    # `group_suffix` marks an ablation of the pipeline itself, so its runs do not
    # land in the same cell as the reference ones when the results are harvested.
    identity = build_run_identity(cfg, dataset=dataset, stage="finetune" if finetune else "pretrain",
                                  extra={"n_modalities": len(modalities)},
                                  group_suffix=getattr(cfg, "group_suffix", None))
    run_name = identity.name
    results_dir = setup_results_dir(cfg, run_name)

    # Trainer + fit
    trainer = instantiate(
        cfg.trainer,
        default_root_dir=results_dir,
        logger=[
            WandbLogger(project=WANDB_PROJECT,
                        name=run_name,
                        save_dir=results_dir,
                        **wandb_resume,
                        **identity.wandb_kwargs())],
        callbacks=callbacks
    )

    if cfg.mode == "train":
        trainer.fit(model, datamodule=data_module, ckpt_path=resume_ckpt)
        # Test the final weights: nothing is selected on the eval split.
        ckpt_path = None
    else:
        ckpt_path = getattr(cfg, "ckpt_path", None)

    trainer.test(model, datamodule=data_module, ckpt_path=ckpt_path)
    wandb.finish()


if __name__ == '__main__':
    main()
