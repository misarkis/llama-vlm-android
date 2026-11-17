#!/bin/bash
# Copyright (c) 2026 Michel Sarkis (https://github.com/misarkis)
#
# This software is released under the MIT License.
# https://opensource.org

# Creates $EVAL/ for the accuracy benchmarks (PC and phone): Python environment and the benchmark
# authors' code at the commits our results were produced with, all under $EVAL_DIR (git-ignored).
#
# Usage (Git Bash, from anywhere):
#   bench/setup_eval.sh                 lmms-eval tasks + ScreenQA-Short
#   bench/setup_eval.sh --mmbench-gui   also MMBench-GUI L1 (second venv + ~2.5 GB of images)
#
# Needs: git, Python 3.11 (tested 3.11.2; PYTHON in .env), and a Hugging Face token in
# HF_TOKEN for the gated lmms-eval datasets. The lmms-eval and ScreenQA datasets download on first
# use; MMBench-GUI data is fetched here.
#
# Creates:
#   $EVAL/.venv                lmms-eval 0.7.4 and the ScreenQA client
#   $EVAL/third_party/         google-research-datasets/screen_qa (official metric),
#                             open-compass/MMBench-GUI + VLMEvalKit (with --mmbench-gui)
#   $EVAL/.venv-vlmeval        VLMEvalKit for MMBench-GUI (with --mmbench-gui)
#   $EVAL/data/                datasets (filled by the harnesses on first use)
#                             data/LMUData_gui198: the 198-question MMBench-GUI phone subset (with --mmbench-gui)
#   $EVAL/logs/ $EVAL/results/  run outputs
#   $EVAL/.setup_done, .setup_mmbench_gui_done   stamps; the run scripts call this script when theirs is missing
set -eu
cd "$(dirname "$0")/.."
# Settings from .env (written by setup-config.sh / setup-config.ps1); a value the caller already set wins
ENV_FILE="$PWD/.env"
if [ -f "$ENV_FILE" ]; then
  while IFS='=' read -r k v || [ -n "$k" ]; do
    v="${v%$'\r'}"
    [[ $k =~ ^[A-Z_][A-Z0-9_]*$ ]] && [ -z "${!k:-}" ] && export "$k=$v"
  done < "$ENV_FILE"
fi
need() { for k; do [ -n "${!k:-}" ] || { echo "$k is not set: run setup-config.sh (it writes .env)"; exit 1; }; done; }
need PYTHON EVAL_DIR
EVAL=$EVAL_DIR   # all generated files go here; the bench/ scripts read the same EVAL_DIR from .env
MMBENCH_GUI=0; [ "${1:-}" = "--mmbench-gui" ] && MMBENCH_GUI=1

mkdir -p $EVAL/third_party $EVAL/data $EVAL/logs $EVAL/results

clone() {   # clone <dir> <url> <commit>: the authors' repo at a fixed commit
  local d=$EVAL/third_party/$1
  [ -d "$d/.git" ] || git clone -q "$2" "$d"
  git -C "$d" checkout -q "$3" 2>/dev/null || { git -C "$d" fetch -q origin; git -C "$d" checkout -q "$3"; }
  echo "$EVAL/third_party/$1 @ $(git -C "$d" rev-parse --short HEAD)"
}
venv() {    # venv <dir>: create if missing, print its python
  [ -d "$1" ] || "$PYTHON" -m venv "$1"
  if [ -x "$1/Scripts/python" ]; then echo "$1/Scripts/python"; else echo "$1/bin/python"; fi
}

# lmms-eval (ocrbench, mmstar, ai2d, chartqa, docvqa_val, textvqa_val, mmmu_val, pope) and ScreenQA-Short
clone screen_qa https://github.com/google-research-datasets/screen_qa 1dfdbccaf56948821b5fa8ffe5d186fe4751e46d
PY=$(venv $EVAL/.venv)
"$PY" -m pip install -q --upgrade pip
"$PY" -m pip install -q lmms-eval==0.7.4 datasets==5.2.0 huggingface_hub==2.4.0 openai==3.50.0 \
    pillow==13.0.0 requests==2.34.2 pyarrow==26.0.1 torch==2.15.0 transformers==5.23.0
echo "$EVAL/.venv: $("$PY" -m pip show lmms-eval | grep ^Version)"

if [ $MMBENCH_GUI = 1 ]; then
  clone MMBench-GUI https://github.com/open-compass/MMBench-GUI.git b67cae38a89916304cba2368ce957d5664857cab
  clone VLMEvalKit https://github.com/open-compass/VLMEvalKit.git 60c47d614b8ecd662ab1a5966c79c778e138b1e3
  PYV=$(venv $EVAL/.venv-vlmeval)
  "$PYV" -m pip install -q --upgrade pip
  "$PYV" -m pip install -q torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
  # MMBench-GUI requirements without flash-attn (GPU only) and VLMEvalKit (installed from the pinned clone)
  grep -vE '^(flash-attn|git\+)' $EVAL/third_party/MMBench-GUI/requirements.txt > $EVAL/third_party/mmbench_gui_requirements.txt
  "$PYV" -m pip install -q -r $EVAL/third_party/mmbench_gui_requirements.txt
  PYTHONUTF8=1 "$PYV" -m pip install -q -e $EVAL/third_party/VLMEvalKit
  # Data: OpenGVLab/MMBench-GUI on Hugging Face (L1 annotations + offline images)
  EVAL=$EVAL "$PYV" - <<'EOF'
import os, zipfile, pathlib
from huggingface_hub import hf_hub_download
out = pathlib.Path(os.environ["EVAL"], "data", "LMUData", "MMBench-GUI")
for f in ["L1_annotations.json", "MMBench-GUI-OfflineImages.zip"]:
    hf_hub_download("OpenGVLab/MMBench-GUI", f, repo_type="dataset", local_dir=out)
if not (out / "offline_images").exists():
    zipfile.ZipFile(out / "MMBench-GUI-OfflineImages.zip").extractall(out)
print(out, "ready")
# Phone subset (bench/run_evals.sh, STEPS gui): every 18th screenshot of L1 = 66 screenshots x 3 questions = 198,
# in LMUData_gui198/ with a link to the same images
import json
sub = out.parent.parent / "LMUData_gui198" / "MMBench-GUI"
sub.mkdir(parents=True, exist_ok=True)
ann = json.loads((out / "L1_annotations.json").read_text(encoding="utf-8"))
(sub / "L1_annotations.json").write_text(json.dumps(ann[::18], ensure_ascii=False), encoding="utf-8")
link = sub / "offline_images"
if not os.path.lexists(link):
    if os.name == "nt":
        import _winapi   # a junction needs no admin rights, unlike a symlink
        _winapi.CreateJunction(str(out / "offline_images"), str(link))
    else:
        os.symlink(out / "offline_images", link, target_is_directory=True)
print(sub, "ready,", sum(len(e["groups"]) for e in ann[::18]), "questions")
EOF
  touch $EVAL/.setup_mmbench_gui_done   # run_mmbench_gui.sh checks this
fi

[ -n "${HF_TOKEN:-}" ] && echo "HF_TOKEN is set" || echo "HF_TOKEN is not set: the gated lmms-eval datasets need it (export HF_TOKEN=...)"
touch $EVAL/.setup_done   # the run scripts check this; written only when every step above succeeded
echo "$EVAL/ ready"
