import torch
import json
import os
import pytorch_lightning as pl

from prefigure.prefigure import get_all_args, push_wandb_config
from stable_audio_tools.data.dataset import create_dataloader_from_config
from stable_audio_tools.models import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict, remove_weight_norm_from_model
from stable_audio_tools.training import create_training_wrapper_from_config, create_demo_callback_from_config
from stable_audio_tools.training.utils import copy_state_dict
from stable_audio_tools.loraw.network import create_lora_from_config
from stable_audio_tools.loraw.callbacks import LoRAModelCheckpoint, ReLoRAModelCheckpoint
from pytorch_lightning.plugins import BitsandbytesPrecisionPlugin

from pytorch_lightning.strategies import DDPStrategy

class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f'{type(err).__name__}: {err}')

class ModelConfigEmbedderCallback(pl.Callback):
    def __init__(self, model_config):
        self.model_config = model_config

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["model_config"] = self.model_config

def main():
    torch.multiprocessing.set_sharing_strategy('file_system')
    args = get_all_args()
    pl.seed_everything(args.seed, workers=True)

    with open(args.model_config) as f:
        model_config = json.load(f)
    with open(args.dataset_config) as f:
        dataset_config = json.load(f)

    train_dl = create_dataloader_from_config(
        dataset_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_rate=model_config["sample_rate"],
        sample_size=model_config["sample_size"],
        audio_channels=model_config.get("audio_channels", 2),
    )

    model = create_model_from_config(model_config)

    if args.pretrained_ckpt_path:
        copy_state_dict(model, load_ckpt_state_dict(args.pretrained_ckpt_path))

    # Checkpoint 로드
    if args.ckpt_path:
        ckpt = torch.load(args.ckpt_path, map_location="cpu")
        if "state_dict" not in ckpt:
            raise KeyError(f"Checkpoint does not contain 'state_dict'. Available keys: {ckpt.keys()}")

        state_dict = ckpt["state_dict"]
        model.load_state_dict(ckpt["state_dict"], strict=False)

    if args.use_lora == 'true':
        lora = create_lora_from_config(model_config, model, use_lisa=True)
        if args.lora_ckpt_path:
            lora.lora_weights(torch.load(args.lora_ckpt_path, map_location="cpu")["state_dict"])
        lora.activate()
        lora.apply_lisa_strategy()

    # 🔹 LISA 적용 이후, Hook 추가 (여기 추가!)
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.register_hook(lambda grad: grad if grad is not None else torch.zeros_like(param))
    
    training_wrapper = create_training_wrapper_from_config(model_config, model)

    if args.use_lora == 'true':
        lora.prepare_for_training(training_wrapper)

    exc_callback = ExceptionCallback()
    logger = pl.loggers.WandbLogger(project=args.name) if args.logger == 'wandb' else None
    checkpoint_dir = args.save_dir if args.save_dir else None

    ckpt_callback = [
        LoRAModelCheckpoint(lora=lora, every_n_train_steps=args.checkpoint_every, dirpath=checkpoint_dir, save_top_k=-1)
    ] if args.use_lora == 'true' else [
        pl.callbacks.ModelCheckpoint(every_n_train_steps=args.checkpoint_every, dirpath=checkpoint_dir, save_top_k=-1)
    ]
    
    demo_callback = create_demo_callback_from_config(model_config, demo_dl=train_dl)
    save_model_config_callback = ModelConfigEmbedderCallback(model_config)

    # 각 GPU 정보 출력
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    trainer = pl.Trainer(
        devices="auto",
        accelerator="gpu",
        strategy=DDPStrategy(find_unused_parameters=True) if args.num_gpus > 1 else "auto", #ddp_find_unused_parameters=True, 에서 변경
        precision=args.precision,
        accumulate_grad_batches=args.accum_batches,
        callbacks=[*ckpt_callback, demo_callback, exc_callback, save_model_config_callback],
        logger=logger,
        log_every_n_steps=1,
        max_epochs=10000000,
        default_root_dir=args.save_dir,
        gradient_clip_val=args.gradient_clip_val,
        reload_dataloaders_every_n_epochs=0,
    )

    trainer.fit(training_wrapper, train_dl, ckpt_path=args.ckpt_path if args.ckpt_path else None)

if __name__ == '__main__':
    main()
