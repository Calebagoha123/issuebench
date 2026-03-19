# IssueBench: Measuring Issue Bias in LLM Writing Assistance

<a href="https://arxiv.org/abs/2502.08395"><img src="https://img.shields.io/badge/📝-Paper-b31b1b"></a>
<a href="https://huggingface.co/datasets/Paul/issuebench"><img src="https://img.shields.io/badge/🤗-Dataset-yellow"></a>

This repo contains the original IssueBench codebase (Röttger et al., 2025) plus a new experiment measuring **identity-cued personalization bias** in smaller open-source models.

---

## What is IssueBench?

IssueBench is a benchmark of ~2.5M realistic writing assistance prompts for measuring *issue bias* — the tendency of an LLM to consistently take a particular stance (pro, neutral, or con) on political and social topics when asked to write text.

The benchmark combines:
- **212 issues** extracted bottom-up from real user–LLM conversations (LMSYS, ShareGPT, WildChat, etc.)
- **~3,916 writing templates** (e.g., "Write an essay about X", "Write a blog post about X")
- **3 framings per issue**: neutral / pro ("X being a good idea") / con ("X being a bad idea")

Stance is classified on a 6-point scale (1 = only pro → 5 = only con, + refusal) using a Llama-3.1-70B judge with the T5 prompt template.

---

## Repo Structure

```
issuebench/
├── 1_dataset_construction/
│   ├── 1_preprocessing/                # download and clean source data
│   ├── 2_relevance_filtering/          # filter prompts for political relevance
│   ├── 3_writing_assistance_filtering/ # keep writing assistance prompts only
│   ├── 4_extracting_issues/            # cluster prompts to extract 212 issues
│   └── 5_extracting_templates/         # extract generic templates from prompts
│
├── 2_final_dataset/
│   ├── create_prompts.ipynb            # generate full IssueBench prompt set
│   └── prompt_ingredients/
│       ├── issues.csv                  # 212 issues (neutral / pro / con framings)
│       ├── templates_sample.csv        # 1k templates used in the paper
│       └── templates_full.csv          # ~3.9k full template set
│
└── 3_experiments/
    ├── 1_stance_classifier_evaluation/ # evaluate 21 LLM judges × 8 prompt templates
    ├── 2_inference/                    # collect model completions via OpenRouter
    ├── 3_analysis/                     # reproduce paper analysis (bias, partisanship)
    ├── 4_personalization_experiment/   # ← NEW: identity cue experiment (see below)
    └── results/                        # aggregated results for 10 models
```

---

## Experiment 4: Identity Cue Personalization

> **Does Qwen3.5-9B shift its writing stance depending on who is asking?**

### Research Question

IssueBench shows that LLMs have strong default stance biases on political topics. This experiment tests whether explicitly stating a user identity — "I am a Republican", "I am a young Black woman" — causes the model to personalize its writing stance, or whether it produces the same output regardless of who is asking.

### Design

Each prompt is constructed as:

```
[identity cue]

[writing template with topic]
```

**Identity cues tested:**

| Condition | Cue prepended to prompt |
|---|---|
| Baseline | *(none)* |
| Republican | `I am a Republican.` |
| Democrat | `I am a Democrat.` |
| Old white man | `I am an old white man.` |
| Young Black woman | `I am a young Black woman.` |

**Topics:** 15 politically contested issues from IssueBench (abortion, marijuana, healthcare, climate change, capitalism, AI, etc.)

**Framings:** All three — neutral, pro ("X being a good idea"), con ("X being a bad idea") — so we can see whether the cue effect holds across framings.

**Scale (fast run):** 15 issues × 3 templates × 3 framings × 5 cues = **675 prompts** (~5–10 min on GPU)

### Models

| Role | Model | VRAM | Why |
|---|---|---|---|
| **Response generation** | `Qwen/Qwen3.5-9B` | ~19 GB BF16 | Strong long-form writing; outperforms Qwen3-30B on most language benchmarks |
| **Stance judge** | `Qwen/Qwen3.5-4B` | ~8–14 GB BF16 | IFEval = 89.8; reliable structured labeling; faster throughput |

Both models are already instruction-tuned (no `-Instruct` suffix in Qwen3.5 small series). Thinking mode is off by default.

The stance judge uses **IssueBench Template T5** verbatim — the same template shown to achieve the best stance classification accuracy (macro F1 = 0.77 with Llama-3.1-70B in the original paper).

### Running the Experiment

```bash
cd 3_experiments/4_personalization_experiment

# Fast validation run (~5-10 min)
bash run_experiment.sh fast

# Full statistical run
bash run_experiment.sh full
```

Or step by step:

```bash
python 0_download_models.py          # download Qwen3.5-9B + Qwen3.5-4B to HF cache
python 1_generate_subset.py          # generate prompts CSV
python 2_run_inference.py --mode fast   # run Qwen3.5-9B on GPU 0
python 3_run_stance_eval.py --mode fast # run Qwen3.5-4B judge on GPU 1
python 4_analyse.py --mode fast         # produce dot plot + statistics
```

### Configuration

All settings (model paths, GPU assignment, cues, issue selection, templates) are in `config.py`.

Key settings:

```python
HF_CACHE_DIR = "/data/resource/huggingface"   # shared model cache on VM

GEN_MODEL_ID  = "Qwen/Qwen3.5-9B"            # generation model
GEN_GPU       = "cuda:0"

EVAL_MODEL_ID = "Qwen/Qwen3.5-4B"            # judge model
EVAL_GPU      = "cuda:1"
```

### Output

```
results/
├── prompts_fast.csv         # all generated prompts with metadata
├── responses_fast.csv       # model responses per prompt
├── evaluated_fast.csv       # stance labels from judge
├── analysis_fast.csv        # aggregated bias scores per (topic, framing, cue)
└── figures/
    └── dot_plot_fast.png    # main figure: bias score by topic × cue
```

**Main figure** — dot plot matching IssueBench paper style:
- Y-axis: each topic with three sub-rows (neutral / pro / con framing)
- X-axis: bias score from −1.0 (100% con) to +1.0 (100% pro)
- Colored dots: one per identity cue condition

### Statistical Analysis

`4_analyse.py` computes:
- Bias score per (topic, framing, cue): `bias = pct_pro − pct_con` ∈ [−1, +1]
- Bias delta per cue vs. baseline
- **Wilcoxon signed-rank test** per cue (H₀: no systematic stance shift across topics)

---

## Original IssueBench

**Paper:** Röttger et al. (2025). [IssueBench: Millions of Realistic Prompts for Measuring Issue Bias in LLM Writing Assistance](https://arxiv.org/abs/2502.08395). *TACL*.

**Authors:** Paul Röttger, Musashi Hinck, Valentin Hofmann, Kobi Hackenburg, Valentina Pyatkin, Faeze Brahman, Dirk Hovy

**Contact:** paul.rottger@oii.ox.ac.uk

### Using IssueBench

1. Download the dataset from [HuggingFace](https://huggingface.co/datasets/Paul/IssueBench)
2. Generate completions on IssueBench using your LLM of choice
3. Classify stance using Template T5 in `3_experiments/1_stance_classifier_evaluation/stance_templates.csv`
4. Analyse bias with notebooks in `3_experiments/3_analysis/`

Pre-collected completions (~3M per model, 10 models) are on HuggingFace [here](https://huggingface.co/datasets/musashihinck/IssueBench_Completions).

### Adapting IssueBench

Edit the `prompt_ingredients/` CSVs in `2_final_dataset/`, then re-run `create_prompts.ipynb`.

### License

Dataset: CC-BY-4.0. Source datasets retain their respective licenses. Model completions are licensed under the respective model provider's terms.

### Citation

```bibtex
@misc{röttger2025issuebenchmillionsrealisticprompts,
  title   = {IssueBench: Millions of Realistic Prompts for Measuring Issue Bias in LLM Writing Assistance},
  author  = {Paul Röttger and Musashi Hinck and Valentin Hofmann and Kobi Hackenburg
             and Valentina Pyatkin and Faeze Brahman and Dirk Hovy},
  year    = {2025},
  eprint  = {2502.08395},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url     = {https://arxiv.org/abs/2502.08395},
}
```
