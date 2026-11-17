# Quality Benchmarks

Scripts that produce the accuracy tables in the main [README](../README.md#quality-benchmarks): the
same Q4_K_M GGUF files, run through `llama-server` on a desktop GPU, then on the phones' NPU. Every
benchmark is scored with its official code: the authors' code for MMBench-GUI and ScreenQA-Short,
lmms-eval for the others. Nothing is re-implemented here; the scripts only start the server, call the
official harness and collect its output.

| Benchmark                                                | Harness                                                                                            | Script                           |
| -------------------------------------------------------- | -------------------------------------------------------------------------------------------------- | -------------------------------- |
| MMBench-GUI L1 (GUI content understanding)               | [open-compass/MMBench-GUI](https://github.com/open-compass/MMBench-GUI) on VLMEvalKit              | `run_mmbench_gui.sh` / `.ps1`    |
| ScreenQA-Short (test split)                              | [google-research-datasets/screen_qa](https://github.com/google-research-datasets/screen_qa) metric | `run_screenqa_short.sh` / `.ps1` |
| DocVQA val, ChartQA, OCRBench, TextVQA val, MMStar, POPE | [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) 0.7.4                                   | `run_public.sh` / `.ps1`         |

The authors' repos are cloned at fixed commits, so a rerun uses the same code as the published results.

## Settings

The scripts read `.env` in the repo root, written by `setup-config.sh` / `setup-config.ps1`:

| Key                                    | Used for                                                                                                |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `LLAMA_SERVER`                         | Desktop `llama-server` executable (desktop runs)                                                        |
| `MODEL_DIR`                            | Folder with the GGUF models (same layout as on the phone)                                               |
| `EVAL_DIR`                             | Benchmark workspace: venvs, authors' code, datasets, logs, results (git-ignored, default `public_eval`) |
| `PYTHON`                               | Python 3.11 for the venvs (tested 3.11.2)                                                               |
| `ANDROID_SDK_ROOT`, `ANDROID_NDK_HOME` | `adb` and the OpenMP runtime (phone runs)                                                               |

The lmms-eval datasets are gated on Hugging Face: set `HF_TOKEN` in the environment before a run
(`export HF_TOKEN=...` or `$env:HF_TOKEN = "..."`). Do not put it in `.env`. lmms-eval copies the
token into its `*_results.json` and logs; the scripts replace it with `hf_REDACTED` after every model.

## Setup (once)

```bash
bench/setup_eval.sh                 # lmms-eval + ScreenQA-Short
bench/setup_eval.sh --mmbench-gui   # also MMBench-GUI L1 (second venv, about 2.5 GB of images)
```

PowerShell: `.\bench\setup_eval.ps1 [-MmbenchGui]`. The run scripts call it themselves on first use.

`--mmbench-gui` also writes the phone subset `$EVAL_DIR/data/LMUData_gui198/`: every 18th screenshot
of L1, 198 questions, linked to the same images.

## Desktop Runs

One `llama-server` per model, started and stopped by the script: `-c 32768 -np 4` (8,192 tokens per
request), `-ngl 99`, `--jinja --reasoning off`. Decoding is greedy.

```bash
bench/run_public.sh         [-m "model1 model2"] [-t "task1,task2"] [-l LIMIT] [-o OUT_DIR]
bench/run_screenqa_short.sh [-m "model1 model2"] [-l LIMIT] [-o OUT_DIR]
bench/run_mmbench_gui.sh    [-m "model1 model2"] [-o OUT_DIR]
```

PowerShell: `.\bench\run_public.ps1 [-Models ...] [-Tasks ...] [-Limit N] [-Out DIR]`, and the same for
the other two.

Without options each script runs all models on the full set. Model names:

`qwen35_0.8b qwen38_2b qwen35_2b qwen3vl_2b qwen2vl_2b qwen35_4b lfm25_1.6b smolvlm2_2.2b granite32_2b`

Example, one model:

```bash
bench/run_public.sh -m qwen35_2b -t "docvqa_val,chartqa,ocrbench,textvqa_val,mmstar,pope"
bench/run_screenqa_short.sh -m qwen35_2b
bench/run_mmbench_gui.sh -m qwen35_2b
```

Outputs go to `$EVAL_DIR/results/` (`public/`, `screenqa_short/`, `mmbench_gui/`) and server and
harness logs to `$EVAL_DIR/logs/`. `CTX`, `SLOTS` and `PORT` can be set in the environment.
`LMU_DIR=$EVAL_DIR/data/LMUData_gui198` runs MMBench-GUI on the 198-question subset instead of the full L1.

## Phone Runs

The same three scripts, with `llama-server` on the phone's NPU behind `adb forward`.
`phone_llama_server.sh` stands in for the desktop server: it maps the model paths to the phone,
sets the phone's context size and `--device HTP0`, and waits for the phone to cool down before each
model. The harnesses and scorers run unchanged on the PC.

1. Build llama.cpp for Android (`build-llamacpp-android.sh`) and push the models to
   `/data/local/tmp/llama.cpp/models/` with the same folder names as `MODEL_DIR`.
2. Connect one phone over `adb`.
3. Run a config from `configs/`:

```bash
bench/run_evals.sh fold7_npu=fold7_npu.conf
bench/run_evals.sh flip6_npu=flip6_npu.conf
```

Each `<folder>=<config>` pair writes `bench/Results/<folder>/` and is skipped if that folder exists.
The config sets:

| Key                          | Fold 7                     | Flip 6                                            |
| ---------------------------- | -------------------------- | ------------------------------------------------- |
| `PHONE_CTX`                  | 16384                      | 8192                                              |
| `PHONE_DEVICE`               | HTP0 (1 HTP session)       | HTP0                                              |
| `PHONE_EXTRA`                | `--cache-ram 0 --no-host`  | `--cache-ram 0 --no-host`                         |
| `MODELS`                     | 7 models                   | 6 models (no `qwen35_4b`: does not fit in memory) |
| `PUBLIC_LIMIT` / `SQA_LIMIT` | 50 / 100                   | 50 / 100                                          |
| `STEPS`                      | `public sqa public512 gui` | `public sqa gui`                                  |

`--cache-ram 0` turns off the server's RAM prompt cache, which grew until the Flip 6 rebooted;
`--no-host` avoids "out of IOVA space" on large images. `gui` runs the 198-question MMBench-GUI subset.

A watchdog logs the phone's free memory every 30 s (`mem.log`) and stops the run, writing `ABORTED`,
if the phone disconnects or free memory stays under 500 MB.

## Results

`bench/Results/` keeps the score files of the published runs:

| Folder       | Contents                                                                              |
| ------------ | ------------------------------------------------------------------------------------- |
| `pc/`        | Desktop runs: full sets, plus the 198-question MMBench-GUI subset (`mmbench_gui198/`) |
| `fold7_npu/` | Galaxy Z Fold 7 NPU, phone subset                                                     |
| `flip6_npu/` | Galaxy Z Flip 6 NPU, phone subset                                                     |

Each folder has a `sources.txt` that describes its files. Only the score files (`*_results.json`,
`*_scores.json`) and run configs are committed; per-question outputs (`.jsonl`, `.xlsx`) stay local.

The desktop numbers for the phone comparison use the same questions: the first 50 (lmms-eval) and
100 (ScreenQA-Short) answers of the full desktop runs, and a separate desktop run on the
198-question MMBench-GUI subset.
