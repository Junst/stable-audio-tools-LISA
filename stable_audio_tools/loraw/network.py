import torch
import gc
from torch import nn
from torch import optim
from enum import Enum
import numpy as np

from stable_audio_tools.models.utils import load_ckpt_state_dict
from .modules import LoRALinear, LoRAConv1d
from .util import *
from .attributes import *

class TargetableModules(Enum):
    Linear = LoRALinear
    Conv1d = LoRAConv1d

def scan_model(model, whitelist=None, blacklist=None):
    whitelist = set(whitelist) if whitelist is not None else None
    blacklist = set(blacklist) if blacklist is not None else None
    module_map = {}
    for decendant_name, decendant_module in model.named_modules():
        if decendant_module.__class__.__name__ in TargetableModules.__members__:
            ancestor_set = set(decendant_name.split("."))
            if (
                (whitelist is None or not ancestor_set.isdisjoint(whitelist))
                and (blacklist is None or ancestor_set.isdisjoint(blacklist))
            ):
                ancestor_module = model
                for name in decendant_name.split(".")[:-1]:
                    ancestor_module = ancestor_module._modules[name]
                id = decendant_name.replace(".", "/")
                module_map[id] = {
                    "module": decendant_module,
                    "parent": ancestor_module,
                }
    print(f"Found {len(module_map)} candidates for LoRA replacement")
    return module_map

class LoRANetwork(nn.Module):
    def __init__(self, target_map, multiplier=1.0, lora_dim=16, alpha=16, dropout=None, module_dropout=None, decompose=False):
        super().__init__()
        self.lora_modules = nn.ModuleDict()
        for name, info in target_map.items():
            module = info["module"]
            self.lora_modules[name] = TargetableModules[module.__class__.__name__].value(
                name,
                module,
                multiplier=multiplier,
                lora_dim=lora_dim,
                alpha=alpha,
                dropout=dropout,
                module_dropout=module_dropout,
                decompose=decompose
            )

    def activate(self, target_map):
        for name, module in self.lora_modules.items():
            module.inject(target_map[name]["parent"])
        print(f"Injected {len(self.lora_modules)} LoRA modules into model")

class LoRAWrapper:
    def __init__(
        self, target_model, model_type=None, component_whitelist=None, multiplier=1.0,
        lora_dim=16, alpha=16, dropout=None, module_dropout=None, decompose=False, lr=None,
        is_lisa=True, lisa_activated_layers=4, lisa_interval_steps=100,
    ):
        self.target_model = target_model
        self.is_lisa = is_lisa
        self.lisa_activated_layers = lisa_activated_layers
        self.lisa_interval_steps = lisa_interval_steps
        self.lr = lr  # lr를 저장 (옵티마이저 설정 시 사용)

        self.target_map = scan_model(target_model, whitelist=component_whitelist)
        self.net = LoRANetwork(self.target_map, multiplier, lora_dim, alpha, dropout, module_dropout, decompose)

        self.layers_attribute = self._get_layers_attribute()
        self.total_layers = len(eval(f'self.{self.layers_attribute}'))
        self.active_layers_indices = []

    def _get_layers_attribute(self):
        model_name = self.target_model.__class__.__name__
        if model_name == "DiffusionTransformer":
            return "target_model.transformer.layers"
        elif model_name == "UNetModel":
            return "target_model.unet.down_blocks + target_model.unet.up_blocks"
        elif model_name == "ConditionedDiffusionModelWrapper":
            return "target_model.model.model.transformer.layers"
        else:
            raise ValueError(f"Unsupported model type: {model_name}")

    def apply_lisa_strategy(self):
        if not self.is_lisa:
            return
        self._switch_active_layers()

    def _switch_active_layers(self):
        if not self.is_lisa:
            return

        layers = eval(f'self.{self.layers_attribute}')

        # 🔥 매번 다른 결과를 얻기 위해 시드를 None으로 설정
        np.random.seed(None)  # 실행할 때마다 다른 레이어가 활성화됨
        num_layers = len(layers)  # `self.model.layers`가 아니라 `layers` 사용
        self.active_layers_indices = np.random.choice(num_layers, size=self.lisa_activated_layers, replace=False)

        for idx in self.active_layers_indices:
            for param in layers[idx].parameters():
                param.requires_grad = True

        print(f"LISA 활성화된 레이어: {self.active_layers_indices}", flush=True)

    def activate(self):
        self.net.activate(self.target_map)
        if self.is_lisa:
            self._switch_active_layers()

    def update_lisa_layers(self, global_step):
        if self.is_lisa and global_step % self.lisa_interval_steps == 0:
            self._switch_active_layers()

    def configure_optimizers(self):
        """
        LoRA 관련 파라미터들을 대상으로 하는 옵티마이저를 설정합니다.
        lr 값이 지정되지 않았다면 기본값을 사용합니다.
        """
        lr = self.lr if self.lr is not None else 1e-3
        optimizer = torch.optim.Adam(self.net.lora_modules.parameters(), lr=lr)
        return optimizer

    def prepare_for_training(self, training_wrapper):
        # Option A: 전체 동결 후, LISA 활성화된 레이어만 다시 unfreeze 처리
        if self.is_lisa:
            layers = eval(f'self.{self.layers_attribute}')
            # 전체 target model 파라미터 동결
            for param in self.target_model.parameters():
                param.requires_grad = False
            # LISA로 활성화된 레이어만 unfreeze
            for idx in self.active_layers_indices:
                for param in layers[idx].parameters():
                    param.requires_grad = True
        else:
            for param in self.target_model.parameters():
                param.requires_grad = False

        # LoRA 네트워크 (lora_modules)는 항상 학습하도록 unfreeze
        for param in self.net.lora_modules.parameters():
            param.requires_grad = True

        # training device로 LoRA 네트워크 이동
        self.net.to(device=training_wrapper.device)

        # 옵티마이저 설정을 LoRA용으로 변경
        training_wrapper.configure_optimizers = self.configure_optimizers

        # EMA 모델이 있다면 잘라내기 (diffusion 모델에 한정)
        if hasattr(training_wrapper, 'diffusion_ema') and training_wrapper.diffusion_ema is not None:
            trim_ema(training_wrapper.diffusion, training_wrapper.diffusion_ema)

        self.is_trainable = True

    def save_weights(self, path):
        torch.save(self.net.lora_modules.state_dict(), path)

    def load_weights(self, residual_weights):
        self.net.lora_modules.load_state_dict(residual_weights)


def create_lora_from_config(config, model, use_lisa):
    lora_config = config["lora"]
    lora = LoRAWrapper(
        model,
        model_type=config["model_type"],
        component_whitelist=lora_config.get("component_whitelist"),
        multiplier=lora_config.get("multiplier"),
        lora_dim=lora_config.get("rank"),
        alpha=lora_config.get("alpha"),
        dropout=lora_config.get("dropout"),
        module_dropout=lora_config.get("module_dropout"),
        lr=lora_config.get("lr"),
        decompose=lora_config.get("weight_decompose", False),
        is_lisa=lora_config.get("is_lisa", use_lisa),
        lisa_activated_layers=lora_config.get("lisa_activated_layers", 4),
        lisa_interval_steps=lora_config.get("lisa_interval_steps", 100),
    )
    return lora