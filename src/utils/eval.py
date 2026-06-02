import argparse
import json
import math
import os
import random

import datasets
import torch
from tqdm.auto import tqdm
from .data_utils import get_test_tokens

def eval_ppl(model, tokenizer, datasets=None, seed=0, seqlen=4096):
    torch.random.manual_seed(seed)
    if datasets is None:
        datasets = ['wikitext2', 'c4']

    result_ppl = {}

    for dataset in datasets:
        input_tok = get_test_tokens(
            dataset,
            tokenizer=tokenizer,
            seed=seed,
            seqlen=seqlen,
        )
        nsamples = input_tok.numel() // seqlen
        input_tok = input_tok[0, :(seqlen * nsamples)].view(nsamples, seqlen)

        device = next(model.parameters()).device
        loss_fct = torch.nn.CrossEntropyLoss().to(device)

        acc_loss = 0.0
        model.eval()

        progress = tqdm(range(nsamples), desc=f"PPL ({dataset})")
        with torch.inference_mode():
            for ii in progress:
                input_ids = input_tok[ii, :].to(device).view(1, -1)
                output = model(
                    input_ids,
                    use_cache=False,
                    output_hidden_states=False,
                    output_attentions=False,
                )[0]

                shift_logits = output[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:]

                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.reshape(-1),
                )

                acc_loss += float(loss.item())
                progress.set_description(f"avg_loss = {acc_loss / (ii + 1):.4f}")

        avg_loss = acc_loss / nsamples
        result_ppl[dataset] = float(torch.exp(torch.tensor(avg_loss)).item())

    return result_ppl

def eval_fewshot(model, tokenizer, tasks=None, seed=0, seqlen=4096, num_fewshot=0, batch_size=1):
    torch.random.manual_seed(seed)
    orig_pad_token = tokenizer.pad_token
    tokenizer.pad_token = tokenizer.eos_token

    if tasks is None:
        tasks = ["arc_challenge"]

    try:
        try:
            import lm_eval
            simple_evaluate = getattr(lm_eval, "simple_evaluate", None)
            if simple_evaluate is None:
                from lm_eval import evaluator
                simple_evaluate = evaluator.simple_evaluate

            from lm_eval.models.huggingface import HFLM
        except Exception as exc:
            raise ModuleNotFoundError(
                "lm-evaluation-harness is installed, but its Python API does not match this code. "
                "Install a recent EleutherAI lm-evaluation-harness/lm-eval version, or avoid calling eval_fewshot()."
            ) from exc

        lm_eval_model = HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
        )

        return simple_evaluate(
            model=lm_eval_model,
            tasks=list(tasks),
            batch_size=batch_size,
            num_fewshot=int(num_fewshot),
        )

    finally:
        tokenizer.pad_token = orig_pad_token