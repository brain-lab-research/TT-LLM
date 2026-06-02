from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.unified_decomp_interface import DecompConfig, DecompositionFactory
from src.utils.eval import eval_ppl
from src.utils.experiment_interface import Experiment


class TensorExperiment(Experiment):
    def __init__(
        self,
        name: str,
        model_path: str,
        decomp_config: DecompConfig,
        *,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        eval_datasets: Sequence[str] = ("wikitext2",),
        seqlen: int = 512,
        random_seed: int = 42,
        save_dir: Optional[str] = None,
        save: bool = True,
        logger=None,
    ):
        super().__init__(name, save_dir, save)
        self.model_path = model_path
        self.decomp_config = decomp_config
        self.device = device
        self.dtype = dtype
        self.eval_datasets = list(eval_datasets)
        self.seqlen = seqlen
        self.random_seed = random_seed
        self.logger = logger

        self.results: List[Dict[str, Any]] = []
        self.baseline_ppl: Optional[Dict[str, float]] = None

        self._decomp = DecompositionFactory.create(decomp_config, device, logger)
        self._tokenizer: Optional[AutoTokenizer] = None
        self._hooks: Dict[str, Any] = {}

    def _load_tokenizer(self) -> AutoTokenizer:
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        return self._tokenizer

    def _load_model(self) -> nn.Module:
        return AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            device_map=self.device,
        )

    def _eval_ppl(self, model: nn.Module) -> Dict[str, float]:
        tokenizer = self._load_tokenizer()
        model.eval()
        with torch.no_grad():
            return eval_ppl(
                model,
                tokenizer,
                datasets=self.eval_datasets,
                seqlen=self.seqlen,
                seed=self.random_seed,
            )

    def _cleanup(self, model: nn.Module) -> None:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _log(self, msg: str) -> None:
        if self.logger is not None:
            self.logger.log(msg)
        else:
            print(msg)

    def register_hook(
        self,
        name: str,
        module: nn.Module,
        fn: Callable,
        *,
        hook_type: str = "forward",
    ) -> None:
        """Attach a named forward (or pre-forward) hook to a module.
        """
        if name in self._hooks:
            self._hooks[name].remove()
        if hook_type == "pre":
            handle = module.register_forward_pre_hook(fn)
        else:
            handle = module.register_forward_hook(fn)
        self._hooks[name] = handle

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self._hooks.values():
            handle.remove()
        self._hooks.clear()

    def experiment(self, layer_indices: Sequence[int]) -> List[Dict[str, Any]]:
        """Sweep over layer_indices: apply decomposition one layer at a time.
        For each layer a fresh model copy is loaded, decomposed, and evaluated.
        Results are appended to self.results (so repeated calls accumulate).
        """
        layer_indices = list(layer_indices)

        if self.baseline_ppl is None:
            self._log("Evaluating baseline PPL...")
            baseline_model = self._load_model()
            self.baseline_ppl = self._eval_ppl(baseline_model)
            self._cleanup(baseline_model)
            self._log(f"Baseline: {self.baseline_ppl}")

        for layer_idx in layer_indices:
            self._log(f"Layer {layer_idx} / {max(layer_indices)} ...")
            model = self._load_model()

            model, apply_time = self._decomp._timed_apply(model, layer_idx)

            ppl = self._eval_ppl(model)

            result: Dict[str, Any] = {
                "layer": layer_idx,
                "ppl": ppl,
                "baseline_ppl": self.baseline_ppl,
                "ppl_delta": {k: ppl[k] - self.baseline_ppl[k] for k in ppl},
                "apply_time_s": round(apply_time, 2),
                "config": self._decomp.config_dict(),
            }
            self.results.append(result)

            self._log(
                f"  layer={layer_idx}  ppl={ppl}  "
                f"delta={result['ppl_delta']}  t={apply_time:.1f}s"
            )

            self._cleanup(model)

        if self.save:
            self.saver()

        return self.results

    def saver(self) -> None:
        if self.save_dir is None:
            self._log("save_dir not set, skipping save.")
            return
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        out = {
            "name": self._name,
            "decomp_config": self._decomp.config_dict(),
            "baseline_ppl": self.baseline_ppl,
            "results": self.results,
        }
        path = Path(self.save_dir) / f"{self._name}.json"
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        self._log(f"Results saved to {path}")
