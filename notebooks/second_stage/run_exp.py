#!/usr/bin/env python3
"""
Modular compression experiment runner for LLaMA models with Tucker/TensorLLM decomposition.
"""

import argparse
import gc
import json
import os
import sys
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

# Fix tqdm display
import tqdm as _tqdm_module
_tqdm_module.tqdm = tqdm

# Setup paths
CWD = Path.cwd().resolve()
REPO_ROOT = CWD if (CWD / 'src').exists() else CWD.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.modular_compression_experiment import (
    MODEL_ARCH_REGISTRY,
    ModelArchitectureSpec,
    ModulePlan,
    RecipeSpec,
    RunnerConfig,
    get_num_layers_from_config,
    get_model_architecture,
    load_results_checkpoint,
    make_target_specs,
    run_experiment_grid,
    save_results,
)
from src.head_decomp_methods import *

# Suppress HuggingFace warnings
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

# Pandas display settings
pd.set_option('display.max_colwidth', 260)


class ExperimentConfig:
    """Configuration for compression experiments."""
    
    # Model settings
    LLAMA_MODEL_NAME = 'meta-llama/Llama-2-7b-hf'
    MODEL_DTYPE = torch.float16
    MODEL_STORAGE_BITS = 16
    LORA_STORAGE_BITS = 16
    SCALE_STORAGE_BITS = 16
    
    # Decomposition settings
    RANKS = [64]
    BITWIDTH = 4
    ORDER = 12
    TOKEN_CHUNK_SIZE = 16
    DECOMPOSE_DTYPE = torch.float64
    DENSE_SPARSE_OUTLIER_FRACTION = 5e-7
    
    # TensorLLM settings
    TENSORLLM_STACK_RANK = 2
    TENSORLLM_HEAD_DIM_RANK = 4
    TENSORLLM_TUCKER_TYPE = 'partial_tucker_v5'
    TENSORLLM_ALLOW_GQA_PER_PROJECTION_FALLBACK = True
    
    # Per-head decomposition settings
    PER_HEAD_TT_ORDER = 4
    PER_HEAD_TUCKER_HIDDEN_RANK = 64
    PER_HEAD_TUCKER_HEAD_DIM_RANK = 16
    
    # LoRA defaults (disabled)
    LORA_R = 0
    LORA_ALPHA = 0
    LORA_DROPOUT = 0.0
    LEARNING_RATE = 0
    WEIGHT_DECAY = 0.0
    MAX_STEPS = 0
    GRAD_ACCUM_STEPS = 8
    TRAIN_BATCH_SIZE = 0
    TRAIN_SEQ_LEN = 0
    NUM_TRAIN_SEQUENCES = 0
    
    # Benchmark settings
    BENCHMARK_WITH_DENSE_RECONSTRUCTION = True
    RUN_PPL = True
    PPL_DATASETS = ['wikitext2', 'c4']
    
    RUN_LM_EVAL = False
    LM_EVAL_TASKS = ['arc_challenge', 'winogrande', 'piqa', 'hellaswag', 'openbookqa']
    LM_EVAL_NUM_FEWSHOT = 0
    LM_EVAL_BATCH_SIZE = 1
    LM_EVAL_LIMIT = None
    
    RUN_HOTPOT_EVAL = False
    HOTPOT_MAX_EXAMPLES = None
    HOTPOT_BATCH_SIZE = 16
    HOTPOT_MAX_NEW_TOKENS = 15
    HOTPOT_BEAM = 1
    
    RUN_ACTIVATION_GEOMETRY = True
    GEOMETRY_NUM_SAMPLES = 100
    GEOMETRY_SEED = 42
    
    RUN_GENERATION_EXAMPLES = True
    GENERATION_PROMPTS = [
        'The theory of tensor train decomposition for neural networks suggests that',
        'Apple Inc. is a worldwide tech company because',
        'Summer is hot. Winter is',
        'Sylvester Stallone is best known for',
        'Sharpness-aware minimization is',
    ]
    GENERATION_MAX_NEW_TOKENS = 80
    
    # Sequence length configurations by architecture
    SEQ_LEN_BY_ARCH = {
        "llama2": {
            "ppl_seqlen": 4096,
            "lora_train_seq_len": 512,
            "geometry_seq_len": 1024,
        },
    }


def setup_environment(args: argparse.Namespace) -> None:
    """Setup CUDA and torch environment."""
    # Set CUDA device
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['HF_USER_ACCESS_TOKEN'] = ''
    os.environ['HF_TOKEN'] = ''
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
    if args.device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    
    # Remove conflicting torch logs
    os.environ.pop("TORCH_LOGS", None)
    print(args.device)
    
    # Setup torch
    torch.set_grad_enabled(True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"Using device: {args.device if args.device is not None else 'default'}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")


def create_layer_targets(n_layers: int, start: Optional[int] = None, end: Optional[int] = None) -> List[dict]:
    """Create target specifications for layers to compress.
    
    Args:
        n_layers: Total number of layers in the model
        start: Starting layer index (inclusive), None for all layers
        end: Ending layer index (exclusive), None for all layers
        
    Returns:
        List of target specifications
    """
    targets = []
    
    if start is not None and end is not None:
        layer_range = range(start, end)
    elif start is not None:
        layer_range = range(start, n_layers)
    elif end is not None:
        layer_range = range(0, end)
    else:
        layer_range = range(n_layers)
    
    for i in layer_range:
        targets.append({
            'label': f"llama_all_layer_{i}",
            'layer_indices': [i],
            'groups': ['mha', 'mlp'],
            'known_superweights': {}
        })
    
    return targets


def make_runner_config(
    model_arch_key: str,
    model_name: str,
    output_dir: Path,
    device_id: int = 0,
    decompose_device: str = 'cuda',
) -> RunnerConfig:
    """Create a RunnerConfig for the experiment.
    
    Args:
        model_arch_key: Architecture key (e.g., 'llama2')
        model_name: HuggingFace model name
        output_dir: Directory for outputs
        device_id: CUDA device ID
        decompose_device: Device for decomposition ('cuda' or 'cpu')
        
    Returns:
        RunnerConfig instance
    """
    seq_cfg = ExperimentConfig.SEQ_LEN_BY_ARCH.get(model_arch_key, {})
    device_map = {"": device_id}
    
    return RunnerConfig(
        model_arch_key=model_arch_key,
        model_name=model_name,
        model_dtype=ExperimentConfig.MODEL_DTYPE,
        model_storage_bits=ExperimentConfig.MODEL_STORAGE_BITS,
        lora_storage_bits=ExperimentConfig.LORA_STORAGE_BITS,
        scale_storage_bits=ExperimentConfig.SCALE_STORAGE_BITS,
        device_map=device_map,
        output_dir=output_dir,
        decompose_dtype=ExperimentConfig.DECOMPOSE_DTYPE,
        decompose_device=decompose_device,
        tt_order=ExperimentConfig.ORDER,
        token_chunk_size=ExperimentConfig.TOKEN_CHUNK_SIZE,
        dense_sparse_outlier_fraction=ExperimentConfig.DENSE_SPARSE_OUTLIER_FRACTION,
        tensorllm_stack_rank=ExperimentConfig.TENSORLLM_STACK_RANK,
        tensorllm_head_dim_rank=ExperimentConfig.TENSORLLM_HEAD_DIM_RANK,
        tensorllm_tucker_type=ExperimentConfig.TENSORLLM_TUCKER_TYPE,
        tensorllm_allow_gqa_per_projection_fallback=ExperimentConfig.TENSORLLM_ALLOW_GQA_PER_PROJECTION_FALLBACK,
        
        lora_r=ExperimentConfig.LORA_R,
        lora_alpha=ExperimentConfig.LORA_ALPHA,
        lora_dropout=ExperimentConfig.LORA_DROPOUT,
        lora_lr=ExperimentConfig.LEARNING_RATE,
        lora_weight_decay=ExperimentConfig.WEIGHT_DECAY,
        lora_max_steps=ExperimentConfig.MAX_STEPS,
        lora_grad_accum_steps=ExperimentConfig.GRAD_ACCUM_STEPS,
        lora_train_batch_size=ExperimentConfig.TRAIN_BATCH_SIZE,
        lora_train_seq_len=seq_cfg.get("lora_train_seq_len", ExperimentConfig.TRAIN_SEQ_LEN),
        lora_num_train_sequences=ExperimentConfig.NUM_TRAIN_SEQUENCES,
        
        run_ppl=ExperimentConfig.RUN_PPL,
        ppl_datasets=ExperimentConfig.PPL_DATASETS,
        ppl_seqlen=seq_cfg.get("ppl_seqlen", 4096),
        
        run_lm_eval=ExperimentConfig.RUN_LM_EVAL,
        lm_eval_tasks=ExperimentConfig.LM_EVAL_TASKS,
        lm_eval_num_fewshot=ExperimentConfig.LM_EVAL_NUM_FEWSHOT,
        lm_eval_batch_size=ExperimentConfig.LM_EVAL_BATCH_SIZE,
        lm_eval_limit=ExperimentConfig.LM_EVAL_LIMIT,
        
        run_hotpot=ExperimentConfig.RUN_HOTPOT_EVAL,
        hotpot_max_examples=ExperimentConfig.HOTPOT_MAX_EXAMPLES,
        hotpot_batch_size=ExperimentConfig.HOTPOT_BATCH_SIZE,
        hotpot_max_new_tokens=ExperimentConfig.HOTPOT_MAX_NEW_TOKENS,
        hotpot_beam=ExperimentConfig.HOTPOT_BEAM,
        
        run_activation_geometry=ExperimentConfig.RUN_ACTIVATION_GEOMETRY,
        geometry_num_samples=ExperimentConfig.GEOMETRY_NUM_SAMPLES,
        geometry_seq_len=seq_cfg.get("geometry_seq_len", 1024),
        geometry_seed=ExperimentConfig.GEOMETRY_SEED,
        
        run_generation_examples=ExperimentConfig.RUN_GENERATION_EXAMPLES,
        generation_prompts=ExperimentConfig.GENERATION_PROMPTS,
        generation_max_new_tokens=ExperimentConfig.GENERATION_MAX_NEW_TOKENS,
        
        use_torch_compile_for_benchmarks=True,
        enable_tf32_for_benchmarks=True,
        torch_compile_mode="reduce-overhead",
        torch_compile_fullgraph=False,
        torch_compile_dynamic=None,
    )


def make_16x_recipes(
    quant_bits: int = 4,
    include_lora: bool = True,
    benchmark_with_dense_reconstruction: bool = True,
) -> List[RecipeSpec]:
    """Base Tucker/TT recipes × {quant/unquant} × {lora/no-lora} = up to 16 recipes.

    Matches make_16x_recipes() from the notebook exactly.
    """
    base = [
        ("tucker_mha", [
            ModulePlan("mha", "tensorllm_tucker_mha", quant_method="none"),
        ]),
        ("tucker_mha__tt_mlp", [
            ModulePlan("mha", "tensorllm_tucker_mha", quant_method="none"),
            ModulePlan("mlp", "tt", quant_method="none"),
        ]),
        ("tt_mlp", [
            ModulePlan("mlp", "tt", quant_method="none"),
        ]),
        ("tt_all", [
            ModulePlan("all", "tt", quant_method="none"),
        ]),
    ]

    recipes = []
    for quantized in [False, True]:
        for with_lora in ([False, True] if include_lora else [False]):
            for label, plans in base:
                new_plans = []
                for p in plans:
                    if quantized:
                        qmethod = (
                            "tensorllm_pre_reconstruct_rtn"
                            if p.decomposition_method.startswith("tensorllm")
                            else "rtn_symmetric"
                        )
                        new_plans.append(ModulePlan(
                            p.target,
                            p.decomposition_method,
                            rank=p.rank,
                            quant_method=qmethod,
                            quant_bits=int(quant_bits),
                            params=dict(p.params or {}),
                        ))
                    else:
                        new_plans.append(ModulePlan(
                            p.target,
                            p.decomposition_method,
                            rank=p.rank,
                            quant_method="none",
                            quant_bits=None,
                            params=dict(p.params or {}),
                        ))

                full_label = (
                    label
                    + (f"__rtn{quant_bits}" if quantized else "")
                    + ("__lora" if with_lora else "")
                )
                recipes.append(RecipeSpec(
                    label=full_label,
                    plans=new_plans,
                    with_lora=bool(with_lora),
                    lora_quant_bits=(int(quant_bits) if with_lora and quantized else None),
                    benchmark_with_dense_reconstruction=benchmark_with_dense_reconstruction,
                ))
    return recipes


def make_new_method_recipes(
    benchmark_with_dense_reconstruction: bool = True,
) -> List[RecipeSpec]:
    """New decomposition method recipes matching notebook cell 16."""
    bases = [
        ("tensorllm_mha_sep_last", [
            ModulePlan("mha", "tensorllm_tucker_mha_sep_last", quant_method="none"),
        ]),
        ("tensorllm_mha_sep_last__ffn", [
            ModulePlan("mha", "tensorllm_tucker_mha_sep_last", quant_method="none"),
            ModulePlan("mlp", "tensorllm_tucker_mlp", quant_method="none"),
        ]),
        ("laser_mha", [
            ModulePlan("mha", "laser", rank=1024, quant_method="none"),
        ]),
        ("laser_ffn", [
            ModulePlan("mlp", "laser", rank=1024, quant_method="none"),
        ]),
        ("tt_per_head_mha", [
            ModulePlan(
                "mha", "tt_per_head", quant_method="none",
                params={"tt_order": ExperimentConfig.PER_HEAD_TT_ORDER},
            ),
        ]),
        ("tucker_per_head_mha", [
            ModulePlan(
                "mha", "tucker_per_head", quant_method="none",
                params={
                    "hidden_rank": ExperimentConfig.PER_HEAD_TUCKER_HIDDEN_RANK,
                    "head_dim_rank": ExperimentConfig.PER_HEAD_TUCKER_HEAD_DIM_RANK,
                },
            ),
        ]),
        ("tensorllm_mha", [
            ModulePlan("mha", "tensorllm_tucker_mha", quant_method="none"),
        ]),
        ("tensorllm_ffn", [
            ModulePlan("mlp", "tensorllm_tucker_mlp", quant_method="none"),
        ]),
        ("tensorllm_mha_ffn", [
            ModulePlan("mha", "tensorllm_tucker_mha", quant_method="none"),
            ModulePlan("mlp", "tensorllm_tucker_mlp", quant_method="none"),
        ]),
    ]

    recipes = []
    for label, plans in bases:
        recipes.append(RecipeSpec(
            label=label,
            plans=plans,
            with_lora=False,
            benchmark_with_dense_reconstruction=benchmark_with_dense_reconstruction,
        ))
    return recipes


def create_recipes(benchmark_with_dense_reconstruction: bool = True) -> List[RecipeSpec]:
    """Create full recipe set matching the notebook: make_16x_recipes + make_new_method_recipes."""
    new = make_new_method_recipes(
        benchmark_with_dense_reconstruction=benchmark_with_dense_reconstruction,
    )
    return new


def load_checkpoint(output_dir: Path, output_json: Path) -> Tuple[List[dict], dict, pd.DataFrame, dict]:
    """Load experiment checkpoint if it exists.
    
    Args:
        output_dir: Output directory
        output_json: Main output JSON file
        
    Returns:
        Tuple of (results, loss_histories, old_status_df, old_payload)
    """
    checkpoint_candidates = []
    
    if output_json.exists():
        checkpoint_candidates.append(output_json)
    
    checkpoint_candidates.extend(sorted(output_dir.glob("*__partial_results.json")))
    checkpoint_candidates = [p for p in checkpoint_candidates if p.exists()]
    
    if checkpoint_candidates:
        checkpoint_path = max(checkpoint_candidates, key=lambda p: p.stat().st_mtime)
        print(f"Loading checkpoint: {checkpoint_path}")
        
        results, old_status_df, old_payload = load_results_checkpoint(checkpoint_path)
        loss_histories = {}
        
        print(f"Restored result rows: {len(results)}")
        
        if len(results) > 0:
            results_df = pd.DataFrame(results)
            print("\nLast 20 results:")
            print(results_df[["model_name", "target_set", "method_label", "rank"]].tail(20))
        
        if len(old_status_df):
            print("\nOld status counts:")
            print(old_status_df.groupby("status", dropna=False).size().reset_index(name="count"))
        
        return results, loss_histories, old_status_df, old_payload
    else:
        print("No checkpoint found; starting fresh.")
        return [], {}, pd.DataFrame(), {}


def deduplicate_results(results: List[dict]) -> List[dict]:
    """Remove duplicate results based on resume_key.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        Deduplicated list of results
    """
    results_df = pd.DataFrame(results)
    
    if len(results_df) > 0 and "resume_key" in results_df.columns:
        before = len(results_df)
        results_df["_resume_key_str"] = results_df["resume_key"].apply(
            lambda x: json.dumps(x, sort_keys=True, default=str)
        )
        results_df = (
            results_df
            .drop_duplicates("_resume_key_str", keep="last")
            .drop(columns=["_resume_key_str"])
        )
        after = len(results_df)
        
        if after < before:
            print(f"Removed {before - after} duplicate rows.")
        
        return results_df.to_dict(orient="records")
    
    return results


def run_experiments(
    output_dir: Path,
    start_layer: Optional[int] = None,
    end_layer: Optional[int] = None,
    device_id: int = 0,
    decompose_device: str = 'cuda',
    exp_name: str = 'tensorllm_exp',
) -> None:
    """Run compression experiments.
    
    Args:
        output_dir: Directory for outputs
        start_layer: Starting layer index (inclusive)
        end_layer: Ending layer index (exclusive)
        device_id: CUDA device ID
        decompose_device: Device for decomposition
        exp_name: Experiment name for output files
    """
    print(f"\n{'='*80}")
    print(f"Running TensorLLM Compression Experiments")
    print(f"{'='*80}")
    print(f"Repository root: {REPO_ROOT}")
    print(f"Output directory: {output_dir}")
    print(f"Experiment name: {exp_name}")
    print(f"Layer range: {start_layer} to {end_layer}")
    print(f"Registered architectures: {sorted(MODEL_ARCH_REGISTRY)}")
    print(f"{'='*80}\n")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_dir / f'{exp_name}_results.json'
    
    # Create runner config
    cfg = make_runner_config(
        model_arch_key='llama2',
        model_name=ExperimentConfig.LLAMA_MODEL_NAME,
        output_dir=output_dir,
        device_id=device_id,
        decompose_device=decompose_device,
    )
    
    # Get number of layers
    n_layers = get_num_layers_from_config(cfg.model_name)
    print(f"Model: {cfg.model_name}")
    print(f"Total layers: {n_layers}")
    
    # Create target specifications
    target_sets = create_layer_targets(n_layers, start=start_layer, end=end_layer)
    print(f"Target sets: {len(target_sets)}")
    
    # Create recipes
    recipes = create_recipes(
        benchmark_with_dense_reconstruction=ExperimentConfig.BENCHMARK_WITH_DENSE_RECONSTRUCTION
    )
    print(f"Recipes: {len(recipes)}")
    for recipe in recipes:
        print(f"  - {recipe.label}")
    
    # Load checkpoint if exists
    results, loss_histories, old_status_df, old_payload = load_checkpoint(output_dir, output_json)
    
    # Run experiments
    print(f"\n{'='*80}")
    print("Starting experiment grid...")
    print(f"{'='*80}\n")
    
    results, loss_histories, status_df = run_experiment_grid(
        target_sets=target_sets,
        recipes=recipes,
        ranks=ExperimentConfig.RANKS,
        cfgs=[cfg],
        results=results,
        loss_histories=loss_histories,
    )
    
    # Deduplicate results
    results = deduplicate_results(results)
    results_df = pd.DataFrame(results)
    
    # Display status
    print(f"\n{'='*80}")
    print("Experiment Status")
    print(f"{'='*80}\n")
    
    if len(status_df):
        print("Status counts:")
        print(status_df.groupby('status', dropna=False).size().reset_index(name='count'))
        
        failed_df = status_df[status_df['status'] == 'failed']
        if len(failed_df):
            print("\nFailed runs:")
            print(failed_df)
    
    # Save results
    print(f"\n{'='*80}")
    print("Saving results...")
    print(f"{'='*80}\n")
    
    saved_json, saved_csv = save_results(
        output_json,
        results_df=results_df,
        results=results,
        cfgs=[cfg],
        target_sets=target_sets,
        recipes=recipes,
        status_df=status_df,
    )
    
    print(f"Saved JSON: {saved_json}")
    print(f"Saved CSV:  {saved_csv}")
    
    # Display summary
    if len(results_df) > 0:
        print(f"\n{'='*80}")
        print("Results Summary")
        print(f"{'='*80}\n")
        
        summary_cols = [
            'model_name', 'target_set', 'method_label', 'rank',
            'ppl_wikitext2',
            'affected_modules_compression_ratio',
            'mha_in_affected_blocks_compression_ratio',
            'mlp_in_affected_blocks_compression_ratio',
        ]
        
        available_cols = [col for col in summary_cols if col in results_df.columns]
        if available_cols:
            print(results_df[available_cols].tail(20))
    
    print(f"\n{'='*80}")
    print("Experiment completed successfully!")
    print(f"{'='*80}\n")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run TensorLLM compression experiments on LLaMA models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        '--device',
        type=int,
        default=None,
        help='CUDA device ID (None for default)',
    )
    
    parser.add_argument(
        '--decompose-device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device for decomposition computation',
    )
    
    parser.add_argument(
        '--start-layer',
        type=int,
        default=None,
        help='Starting layer index (inclusive)',
    )
    
    parser.add_argument(
        '--end-layer',
        type=int,
        default=None,
        help='Ending layer index (exclusive)',
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output directory for results (default: REPO_ROOT/results_tensorllm_STARTLAYER_ENDLAYER)',
    )
    
    parser.add_argument(
        '--exp-name',
        type=str,
        default='tensorllm_exp',
        help='Experiment name for output files',
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup environment
    setup_environment(args)
    
    # Determine output directory
    if args.output_dir is None:
        if args.start_layer is not None or args.end_layer is not None:
            start_str = str(args.start_layer) if args.start_layer is not None else '0'
            end_str = str(args.end_layer) if args.end_layer is not None else 'end'
            output_dir = REPO_ROOT / f'results_tensorllm_{start_str}_{end_str}'
        else:
            output_dir = REPO_ROOT / 'results_tensorllm'
    else:
        output_dir = args.output_dir
    
    # Run experiments
    run_experiments(
        output_dir=output_dir,
        start_layer=args.start_layer,
        end_layer=args.end_layer,
        device_id=args.device if args.device is not None else 0,
        decompose_device=args.decompose_device,
        exp_name=args.exp_name,
    )


if __name__ == '__main__':
    main()
