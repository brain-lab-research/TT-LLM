# %% [markdown]
# # Qwen3 TD-MoE Experiments, CR 0.4
#
# Full MoE-only experiment entrypoint for Qwen3-30B-A3B.
# It benchmarks baseline and dense-reconstructed TD-MoE compressed models.

# %%
from pathlib import Path
import json
import os
import sys

import pandas as pd
import torch
from tqdm.auto import tqdm

try:
    from IPython.display import display
except Exception:
    display = print

CWD = Path.cwd().resolve()
REPO_ROOT = CWD if (CWD / "src").exists() else CWD.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

print("Repo root:", REPO_ROOT)
print("Has src:", (REPO_ROOT / "src").exists())

# %%
from src.modular_compression_experiment import (
    MODEL_ARCH_REGISTRY,
    ModulePlan,
    RecipeSpec,
    RunnerConfig,
    get_model_architecture,
    get_num_layers_from_config,
    make_target_specs,
    middle_out_expanding_layer_sets,
    run_experiment_grid,
    save_results,
)

pd.set_option("display.max_colwidth", 260)
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("Registered model architectures:", sorted(MODEL_ARCH_REGISTRY))

# %% [markdown]
# ## Configuration

# %%
MODEL_ARCH_KEY = "qwen3_30b_a3b"
QWEN3_MODEL_NAME = os.environ.get("QWEN3_MODEL_NAME", "Qwen/Qwen3-30B-A3B")

def resolve_output_root() -> Path:
    raw = os.environ.get("SASHA_OUTPUT_ROOT", "outputs")
    path = Path(raw).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


OUTPUT_ROOT = resolve_output_root()
OUTPUT_DIR = OUTPUT_ROOT / "results_qwen3_30b_td_moe_cr0p4"
OUTPUT_JSON = OUTPUT_DIR / "qwen3_30b_td_moe_cr0p4_results.json"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DTYPE = torch.bfloat16
MODEL_STORAGE_BITS = 16
SCALE_STORAGE_BITS = 16
LORA_STORAGE_BITS = 16
DEVICE_MAP = os.environ.get("QWEN3_DEVICE_MAP", "auto")
DECOMPOSE_DTYPE = torch.float32
DECOMPOSE_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

COMPRESSION_RATIOS = [0.4]
EXPERT_MODE = "preserve"
SVD_BACKEND = "gram_eigh"
TD_MOE_GRID_RANKS = [0]  # TD-MoE chooses Tucker ranks from the compression budget.

CENTER_LAYER = None  # None means num_layers // 2.
RIGHT_WARMUP_FRACTION = 0.25
MAX_TARGET_LAYER_FRACTION = 0.5
TARGET_SET_LIMIT = None  # Set to a small integer while debugging.

BENCHMARK_WITH_DENSE_RECONSTRUCTION = True
RUN_PPL = True
PPL_DATASETS = ["wikitext2", "c4"]
PPL_SEQLEN = 4096

RUN_LM_EVAL = True
LM_EVAL_TASKS = ["arc_challenge", "winogrande", "piqa", "hellaswag", "openbookqa"]
LM_EVAL_NUM_FEWSHOT = 0
LM_EVAL_BATCH_SIZE = 1
LM_EVAL_LIMIT = None
LM_EVAL_STRICT = True

RUN_HOTPOT_EVAL = False
HOTPOT_MAX_EXAMPLES = None
HOTPOT_BATCH_SIZE = 16
HOTPOT_MAX_NEW_TOKENS = 15
HOTPOT_BEAM = 1

RUN_ACTIVATION_GEOMETRY = True
GEOMETRY_NUM_SAMPLES = 100
GEOMETRY_SEQ_LEN = 1024
GEOMETRY_SEED = 42

RUN_GENERATION_EXAMPLES = True
GENERATION_PROMPTS = [
    "The theory of tensor train decomposition for neural networks suggests that",
    "Apple Inc. is a worldwide tech company because",
    "Summer is hot. Winter is",
    "Sylvester Stallone is best known for",
    "Sharpness-aware minimization is",
]
GENERATION_MAX_NEW_TOKENS = 80

USE_TORCH_COMPILE_FOR_BENCHMARKS = False
ENABLE_TF32_FOR_BENCHMARKS = True

# %% [markdown]
# ## Runner Config

# %%
assert MODEL_ARCH_KEY in MODEL_ARCH_REGISTRY, sorted(MODEL_ARCH_REGISTRY)
model_spec = get_model_architecture(MODEL_ARCH_KEY)
print(model_spec)

cfg = RunnerConfig(
    model_arch_key=MODEL_ARCH_KEY,
    model_name=QWEN3_MODEL_NAME,
    model_dtype=MODEL_DTYPE,
    model_storage_bits=MODEL_STORAGE_BITS,
    lora_storage_bits=LORA_STORAGE_BITS,
    scale_storage_bits=SCALE_STORAGE_BITS,
    device_map=DEVICE_MAP,
    output_dir=OUTPUT_DIR,
    decompose_dtype=DECOMPOSE_DTYPE,
    decompose_device=DECOMPOSE_DEVICE,
    run_ppl=RUN_PPL,
    ppl_datasets=PPL_DATASETS,
    ppl_seqlen=PPL_SEQLEN,
    run_lm_eval=RUN_LM_EVAL,
    lm_eval_tasks=LM_EVAL_TASKS,
    lm_eval_num_fewshot=LM_EVAL_NUM_FEWSHOT,
    lm_eval_batch_size=LM_EVAL_BATCH_SIZE,
    lm_eval_limit=LM_EVAL_LIMIT,
    lm_eval_strict=LM_EVAL_STRICT,
    run_hotpot=RUN_HOTPOT_EVAL,
    hotpot_max_examples=HOTPOT_MAX_EXAMPLES,
    hotpot_batch_size=HOTPOT_BATCH_SIZE,
    hotpot_max_new_tokens=HOTPOT_MAX_NEW_TOKENS,
    hotpot_beam=HOTPOT_BEAM,
    run_activation_geometry=RUN_ACTIVATION_GEOMETRY,
    geometry_num_samples=GEOMETRY_NUM_SAMPLES,
    geometry_seq_len=GEOMETRY_SEQ_LEN,
    geometry_seed=GEOMETRY_SEED,
    run_generation_examples=RUN_GENERATION_EXAMPLES,
    generation_prompts=GENERATION_PROMPTS,
    generation_max_new_tokens=GENERATION_MAX_NEW_TOKENS,
    use_torch_compile_for_benchmarks=USE_TORCH_COMPILE_FOR_BENCHMARKS,
    enable_tf32_for_benchmarks=ENABLE_TF32_FOR_BENCHMARKS,
)

RUNNER_CONFIGS = [cfg]
display(pd.DataFrame([{
    "model_arch_key": c.model_arch_key,
    "model_name": c.model_name,
    "dtype": str(c.model_dtype),
    "device_map": c.device_map,
    "decompose_device": c.decompose_device,
    "ppl_datasets": c.ppl_datasets,
    "ppl_seqlen": c.ppl_seqlen,
    "run_lm_eval": c.run_lm_eval,
    "lm_eval_tasks": c.lm_eval_tasks,
    "lm_eval_limit": c.lm_eval_limit,
} for c in RUNNER_CONFIGS]))

# %% [markdown]
# ## Target Layer Sets

# %%
n_layers = get_num_layers_from_config(QWEN3_MODEL_NAME)
TARGET_SETS = middle_out_expanding_layer_sets(
    n_layers,
    label_prefix="qwen3_30b_td_moe_middle_out",
    center_layer=CENTER_LAYER,
    right_warmup_fraction=RIGHT_WARMUP_FRACTION,
    max_fraction=MAX_TARGET_LAYER_FRACTION,
    groups=["moe_experts"],
)
if TARGET_SET_LIMIT is not None:
    TARGET_SETS = TARGET_SETS[: int(TARGET_SET_LIMIT)]

TARGET_SETS_BY_MODEL = {QWEN3_MODEL_NAME: TARGET_SETS}
print("num_layers:", n_layers)
display(pd.DataFrame(TARGET_SETS))

# %% [markdown]
# ## Recipes

# %%
def ratio_label(value: float) -> str:
    return str(value).replace(".", "p")


RECIPES = [
    RecipeSpec(
        label=f"td_moe_gate_up_cr{ratio_label(ratio)}",
        plans=[
            ModulePlan(
                "moe_experts",
                "td_moe",
                rank=None,
                quant_method="none",
                params={
                    "compression_ratio": float(ratio),
                    "expert_mode": EXPERT_MODE,
                    "svd_backend": SVD_BACKEND,
                },
            )
        ],
        with_lora=False,
        benchmark_with_dense_reconstruction=BENCHMARK_WITH_DENSE_RECONSTRUCTION,
    )
    for ratio in COMPRESSION_RATIOS
]

display(pd.DataFrame([{
    "label": r.label,
    "with_lora": r.with_lora,
    "benchmark_with_dense_reconstruction": r.benchmark_with_dense_reconstruction,
    "plans": [p.__dict__ for p in r.plans],
} for r in RECIPES]))

# %% [markdown]
# ## Sanity Check Target Expansion

# %%
for target_set in TARGET_SETS[: min(3, len(TARGET_SETS))]:
    print("TARGET SET:", target_set["label"])
    display(pd.DataFrame(make_target_specs(target_set, model_spec)))

# %% [markdown]
# ## Run Experiments
#
# The JSON output is loaded first, so reruns skip completed rows.

# %%
if OUTPUT_JSON.exists():
    with open(OUTPUT_JSON) as f:
        previous_payload = json.load(f)
    results = list(previous_payload.get("results", []))
else:
    results = []

loss_histories = {}
all_status = []

pb = tqdm(RUNNER_CONFIGS)
for run_cfg in pb:
    pb.set_description(f"Model {run_cfg.model_name}")
    target_sets = TARGET_SETS_BY_MODEL[run_cfg.model_name]
    results, loss_histories, status_df = run_experiment_grid(
        target_sets=target_sets,
        recipes=RECIPES,
        ranks=TD_MOE_GRID_RANKS,
        cfgs=[run_cfg],
        results=results,
        loss_histories=loss_histories,
    )
    all_status.append(status_df)

run_status_df = pd.concat(all_status, ignore_index=True) if all_status else pd.DataFrame()
display(run_status_df)
if len(run_status_df):
    display(run_status_df.groupby("status", dropna=False).size().reset_index(name="count"))

# %% [markdown]
# ## Results

# %%
results_df = pd.DataFrame(results)
if len(results_df) > 0 and "resume_key" in results_df.columns:
    before = len(results_df)
    results_df["_resume_key_str"] = results_df["resume_key"].apply(lambda x: json.dumps(x, sort_keys=True, default=str))
    results_df = results_df.drop_duplicates("_resume_key_str", keep="last").drop(columns=["_resume_key_str"])
    after = len(results_df)
    if after < before:
        print(f"Removed {before - after} duplicate rows.")
        results = results_df.to_dict(orient="records")

display(results_df)

compact_cols = [
    "model_name",
    "target_set",
    "method_label",
    "target_layers",
    "ppl_wikitext2",
    "ppl_c4",
    "affected_modules_compression_ratio",
    "moe_experts_in_affected_blocks_compression_ratio",
    "affected_blocks_compression_ratio",
    "overall_no_embeddings_compression_ratio",
    "overall_with_embeddings_compression_ratio",
    "generation_preview_text",
]
display(results_df[[c for c in compact_cols if c in results_df.columns]])

# %% [markdown]
# ## Save

# %%
saved_json, saved_csv = save_results(
    OUTPUT_JSON,
    results_df=results_df,
    results=results,
    cfgs=RUNNER_CONFIGS,
    target_sets=TARGET_SETS,
    recipes=RECIPES,
    status_df=run_status_df if "run_status_df" in globals() else None,
)
print("Saved JSON:", saved_json)
print("Saved CSV :", saved_csv)
