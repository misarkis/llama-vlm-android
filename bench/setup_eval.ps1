# Copyright (c) 2026 Michel Sarkis (https://github.com/misarkis)
#
# This software is released under the MIT License.
# https://opensource.org

# =============================================================================
# setup_eval.ps1
# =============================================================================
# Creates $EVAL_DIR\ for the accuracy benchmarks (PC and phone): Python environment and the benchmark
# authors' code at the commits our results were produced with, all under $EVAL_DIR (git-ignored).
# PowerShell version of bench/setup_eval.sh (same pins, same layout).
#
# SETTINGS (.env, written by setup-config.ps1): PYTHON, EVAL_DIR
#
# USAGE (from anywhere):
#   .\bench\setup_eval.ps1               lmms-eval tasks + ScreenQA-Short
#   .\bench\setup_eval.ps1 -MmbenchGui   also MMBench-GUI L1 (second venv + ~2.5 GB of images)
#
# Needs: git, Python 3.11 (tested 3.11.2; PYTHON in .env), and a Hugging Face token in
# $env:HF_TOKEN for the gated lmms-eval datasets. The lmms-eval and ScreenQA datasets download on first
# use; MMBench-GUI data is fetched here.
#
# Creates:
#   $EVAL_DIR\.venv                lmms-eval 0.7.4 and the ScreenQA client
#   $EVAL_DIR\third_party\         google-research-datasets/screen_qa (official metric),
#                                  open-compass/MMBench-GUI + VLMEvalKit (with -MmbenchGui)
#   $EVAL_DIR\.venv-vlmeval        VLMEvalKit for MMBench-GUI (with -MmbenchGui)
#   $EVAL_DIR\data\                datasets (filled by the harnesses on first use)
#                                  data\LMUData_gui198: the 198-question MMBench-GUI phone subset (with -MmbenchGui)
#   $EVAL_DIR\logs\ results\       run outputs
#   $EVAL_DIR\.setup_done, .setup_mmbench_gui_done   stamps; the run scripts call this script when theirs is missing
# =============================================================================

param([switch]$MmbenchGui)

. (Join-Path $PSScriptRoot "bench_common.ps1")
Assert-Settings @("PYTHON", "EVAL_DIR")

$Eval = $env:EVAL_DIR   # all generated files go here; the bench\ scripts read the same EVAL_DIR from .env

# Native commands do not stop the script on failure by themselves; this does
function Invoke-Native {
    param([scriptblock]$Command)
    & $Command
    if ($LASTEXITCODE -ne 0) { Write-Host "failed (exit $LASTEXITCODE): $Command" -ForegroundColor Red; exit 1 }
}

New-Item -ItemType Directory -Force -Path "$Eval\third_party", "$Eval\data", "$Eval\logs", "$Eval\results" | Out-Null

# The authors' repo at a fixed commit
function Get-PinnedRepo {
    param([string]$Dir, [string]$Url, [string]$Commit)
    $d = "$Eval\third_party\$Dir"
    if (-not (Test-Path "$d\.git")) { Invoke-Native { git clone -q $Url $d } }
    git -C $d checkout -q $Commit 2>$null
    if ($LASTEXITCODE -ne 0) {
        Invoke-Native { git -C $d fetch -q origin }
        Invoke-Native { git -C $d checkout -q $Commit }
    }
    Write-Host "$Eval\third_party\$Dir @ $(git -C $d rev-parse --short HEAD)"
}

# Create the venv if missing and return its python
function New-Venv {
    param([string]$Dir)
    if (-not (Test-Path $Dir)) { Invoke-Native { & $env:PYTHON -m venv $Dir } }
    return Get-VenvPython $Dir
}

# lmms-eval (ocrbench, mmstar, ai2d, chartqa, docvqa_val, textvqa_val, mmmu_val, pope) and ScreenQA-Short
Get-PinnedRepo "screen_qa" "https://github.com/google-research-datasets/screen_qa" "1dfdbccaf56948821b5fa8ffe5d186fe4751e46d"
$Py = New-Venv "$Eval\.venv"
Invoke-Native { & $Py -m pip install -q --upgrade pip }
Invoke-Native { & $Py -m pip install -q lmms-eval==0.7.4 datasets==5.2.0 huggingface_hub==2.4.0 openai==3.50.0 `
    pillow==13.0.0 requests==2.34.2 pyarrow==26.0.1 torch==2.15.0 transformers==5.23.0 }
Write-Host "$Eval\.venv: $(& $Py -m pip show lmms-eval | Select-String '^Version')"

if ($MmbenchGui) {
    Get-PinnedRepo "MMBench-GUI" "https://github.com/open-compass/MMBench-GUI.git" "b67cae38a89916304cba2368ce957d5664857cab"
    Get-PinnedRepo "VLMEvalKit" "https://github.com/open-compass/VLMEvalKit.git" "60c47d614b8ecd662ab1a5966c79c778e138b1e3"
    $PyV = New-Venv "$Eval\.venv-vlmeval"
    Invoke-Native { & $PyV -m pip install -q --upgrade pip }
    Invoke-Native { & $PyV -m pip install -q torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu }
    # MMBench-GUI requirements without flash-attn (GPU only) and VLMEvalKit (installed from the pinned clone)
    $req = "$Eval\third_party\mmbench_gui_requirements.txt"
    Get-Content "$Eval\third_party\MMBench-GUI\requirements.txt" |
        Where-Object { $_ -notmatch '^(flash-attn|git\+)' } | Set-Content -Path $req -Encoding utf8NoBOM
    Invoke-Native { & $PyV -m pip install -q -r $req }
    $env:PYTHONUTF8 = "1"
    Invoke-Native { & $PyV -m pip install -q -e "$Eval\third_party\VLMEvalKit" }
    # Data: OpenGVLab/MMBench-GUI on Hugging Face (L1 annotations + offline images)
    $env:EVAL = $Eval
    $fetch = @'
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
'@
    Invoke-Native { $fetch | & $PyV - }
    New-Item -ItemType File -Force -Path "$Eval\.setup_mmbench_gui_done" | Out-Null   # run_mmbench_gui.ps1 checks this
}

if ($env:HF_TOKEN) { Write-Host "HF_TOKEN is set" }
else { Write-Host "HF_TOKEN is not set: the gated lmms-eval datasets need it (`$env:HF_TOKEN = '...')" }
New-Item -ItemType File -Force -Path "$Eval\.setup_done" | Out-Null   # the run scripts check this; written only when every step above succeeded
Write-Host "$Eval\ ready"
