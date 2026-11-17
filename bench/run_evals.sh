#!/bin/bash
# Copyright (c) 2026 Michel Sarkis (https://github.com/misarkis)
#
# This software is released under the MIT License.
# https://opensource.org

# Accuracy benchmarks on the phone, the eval counterpart of profiler/run_profiles.sh.
# Runs the unchanged harnesses (bench/run_public.sh -> lmms-eval, bench/run_screenqa_short.sh -> Google's
# ScreenQA metric, bench/run_mmbench_gui.sh -> the MMBench-GUI authors' code)
# against llama-server on the phone's NPU (bench/phone_llama_server.sh), one model at a time.
#
# Usage (from anywhere, Git Bash):
#   bench/run_evals.sh <folder>=<config> ...
#   config: a file in bench/configs/ (PHONE_CTX, PHONE_DEVICE, PHONE_EXTRA, MODELS, PUBLIC_LIMIT, SQA_LIMIT, STEPS)
#   STEPS: any of public, sqa, public512, gui (default "public sqa"). public512 reruns mmmu_val with
#   max_new_tokens=512, as the PC rerun does: at the lmms-eval default (128) many answers are cut off.
#   gui runs MMBench-GUI L1 on the 198-question subset in $EVAL_DIR/data/LMUData_gui198 (every 18th
#   screenshot; made by bench/setup_eval.sh --mmbench-gui).
#   e.g. bench/run_evals.sh flip6_npu=flip6_npu.conf
#
# Each <folder> goes to bench/Results/<folder>/ (skipped if it exists):
#   <config>                    copy of the config used
#   public/ screenqa_short/     harness outputs (samples + scores)
#   public512/                  mmmu_val at max_new_tokens=512 (STEPS public512)
#   mmbench_gui/                MMBench-GUI L1 subset predictions + scores (STEPS gui)
#   logs/                       phone server, lmms-eval and ScreenQA logs
#   run_settings.txt            phone model, build commit, start time
#   script_changes.diff         the only differences of the run copies from bench/run_*.sh (log paths, client concurrency)
#   mem.log                     adb state and phone MemAvailable every 30 s
#   ABORTED                     written if the phone disappears or runs out of memory; the run stops there
cd "$(dirname "$0")/.." || exit 1
REPO=$PWD
# Settings from .env (written by setup-config.sh / setup-config.ps1); a value the caller already set wins
ENV_FILE="$REPO/.env"
if [ -f "$ENV_FILE" ]; then
  while IFS='=' read -r k v || [ -n "$k" ]; do
    v="${v%$'\r'}"
    [[ $k =~ ^[A-Z_][A-Z0-9_]*$ ]] && [ -z "${!k:-}" ] && export "$k=$v"
  done < "$ENV_FILE"
fi
need() { for k; do [ -n "${!k:-}" ] || { echo "$k is not set: run setup-config.sh (it writes .env)"; exit 1; }; done; }
need ANDROID_SDK_ROOT MODEL_DIR EVAL_DIR
export MSYS_NO_PATHCONV=1
ADB="$ANDROID_SDK_ROOT/platform-tools/adb"
export PATH="$(dirname "$ADB"):$PATH"
RESULTS=${RESULTS_DIR:-bench/Results}
MEM_FLOOR_MB=${MEM_FLOOR_MB:-500}
export REPO_ROOT=$REPO LLAMA_SERVER=$REPO/bench/phone_llama_server.sh PORT=${PORT:-8090} SLOTS=1 CTX=4096
DEV_SERVER=/data/local/tmp/llama-server-eval/llama-server
mkdir -p "$RESULTS"

# Run copies of the bench/run_*.sh scripts: only log paths and the client's concurrency/timeout differ
make_copies() {
  local name=$1 logs=$1/logs
  # ROOT is the working directory: run_evals.sh runs the copies from the repo root (no absolute paths in results)
  sed -e 's|^ROOT=.*|ROOT="$PWD"|' \
      -e "s|\\\$EVAL/logs/server_|$logs/server_|; s|\\\$EVAL/logs/lmms_|$logs/lmms_|" \
      -e 's|api_key=EMPTY"|api_key=EMPTY,num_concurrent=1,timeout=600"|' \
      bench/run_public.sh > "$name/run_public_phone.sh"
  sed -e 's|^ROOT=.*|ROOT="$PWD"|' \
      -e "s|\\\$EVAL/logs/server_|$logs/server512_|; s|\\\$EVAL/logs/lmms_|$logs/lmms512_|" \
      -e 's|api_key=EMPTY"|api_key=EMPTY,num_concurrent=1,timeout=600"|' \
      -e 's|--batch_size 1 --log_samples|--batch_size 1 --gen_kwargs max_new_tokens=512 --log_samples|' \
      bench/run_public.sh > "$name/run_public512_phone.sh"
  sed -e 's|^ROOT=.*|ROOT="$PWD"|' \
      -e "s|\\\$EVAL/logs/server_sqa_|$logs/server_sqa_|; s|\\\$EVAL/logs/sqa_|$logs/sqa_|" \
      bench/run_screenqa_short.sh > "$name/run_screenqa_phone.sh"
  sed -e 's|^ROOT=.*|ROOT="$PWD"|' \
      -e "s|\\\$EVAL/logs/server_gui_|$logs/server_gui_|; s|\\\$EVAL/logs/gui_|$logs/gui_|" \
      bench/run_mmbench_gui.sh > "$name/run_mmbench_gui_phone.sh"
  { diff bench/run_public.sh "$name/run_public_phone.sh"; diff bench/run_screenqa_short.sh "$name/run_screenqa_phone.sh"
    diff bench/run_public.sh "$name/run_public512_phone.sh"; diff bench/run_mmbench_gui.sh "$name/run_mmbench_gui_phone.sh"; } \
      > "$name/script_changes.diff"
  # 5 changed lines in run_public.sh (ROOT, 3 log paths, client options), 4 in run_screenqa_short.sh
  # (ROOT, 3 log paths), 6 in the 512 copy (the 5 + gen_kwargs), 4 in run_mmbench_gui.sh (ROOT, 3 log paths);
  # anything else means bench/run_*.sh changed shape
  local n; n=$(grep -c '^>' "$name/script_changes.diff")
  if [ "$n" -ne 19 ]; then echo "script copies changed $n lines, expected 19; see $name/script_changes.diff"; return 1; fi
}

# Stops the run if the phone disappears or memory stays below the floor (the Flip6 rebooted under memory pressure)
watchdog() {
  local name=$1 lost=0 low=0 st mem pid
  while [ ! -e "$name/.done" ]; do
    st=$("$ADB" get-state 2>/dev/null | tr -d '\r')
    mem=$("$ADB" shell "grep MemAvailable /proc/meminfo" 2>/dev/null | awk '{print int($2/1024)}')
    echo "$(date +%T) state=${st:-none} mem_avail_mb=${mem:-?}" >> "$name/mem.log"
    if [ "$st" = device ]; then lost=0; else lost=$((lost + 1)); fi
    if [ -n "$mem" ] && [ "$mem" -lt "$MEM_FLOOR_MB" ]; then low=$((low + 1)); else low=0; fi
    if [ $lost -ge 2 ] || [ $low -ge 2 ]; then
      echo "$(date +%T) state=${st:-none} mem_avail_mb=${mem:-?} (floor $MEM_FLOOR_MB)" > "$name/ABORTED"
      pid=$(cat "$name/.step_pid" 2>/dev/null)
      # single-slash flags: with MSYS_NO_PATHCONV=1, //F reaches taskkill as is and is rejected
      [ -n "$pid" ] && [ -e "/proc/$pid/winpid" ] && taskkill /F /T /PID "$(cat "/proc/$pid/winpid")" >/dev/null 2>&1
      "$ADB" shell "pkill -f $DEV_SERVER" >/dev/null 2>&1
      return
    fi
    sleep 30
  done
}

step() {   # step <folder> <command...>: run one harness call, killable by the watchdog
  local name=$1; shift
  "$@" >> "$name/console.txt" 2>&1 &
  echo $! > "$name/.step_pid"
  wait $!
}

for pair in "$@"; do
  name=$RESULTS/${pair%%=*}; config=${pair#*=}
  if [ ! -f "bench/configs/$config" ]; then echo "SKIP $name: bench/configs/$config not found"; continue; fi
  if [ -e "$name" ]; then echo "SKIP $name: folder exists"; continue; fi
  if [ "$("$ADB" devices | grep -c 'device$')" -ne 1 ]; then echo "need exactly one phone on adb"; exit 1; fi
  unset PHONE_CTX PHONE_DEVICE PHONE_EXTRA MODELS PUBLIC_LIMIT SQA_LIMIT STEPS
  set -a; . "bench/configs/$config"; set +a
  for v in PHONE_CTX PHONE_DEVICE MODELS PUBLIC_LIMIT SQA_LIMIT; do
    if [ -z "${!v:-}" ]; then echo "SKIP $name: $v missing in bench/configs/$config"; continue 2; fi
  done

  mkdir -p "$name/logs"
  cp "bench/configs/$config" "$name/"
  make_copies "$name" || exit 1
  {
    echo "phone_model=$("$ADB" shell getprop ro.product.model | tr -d '\r')"
    echo "llama.cpp=$(git -C llama.cpp rev-parse --short HEAD 2>/dev/null)"
    echo "started=$(date '+%F %T')"
  } > "$name/run_settings.txt"
  echo "=== $(date +%T) START $name ($config, ctx $PHONE_CTX)"
  bash bench/phone_llama_server.sh --setup >> "$name/console.txt" 2>&1
  watchdog "$name" &
  wd=$!

  for m in $MODELS; do
    for s in ${STEPS:-public sqa}; do
      [ -e "$name/ABORTED" ] && break 2
      case $s in
        public)    echo "=== $(date +%T) $name $m: public x$PUBLIC_LIMIT"
                   step "$name" bash "$name/run_public_phone.sh" -m "$m" -l "$PUBLIC_LIMIT" -o "$name/public" ;;
        sqa)       echo "=== $(date +%T) $name $m: screenqa x$SQA_LIMIT"
                   step "$name" bash "$name/run_screenqa_phone.sh" -m "$m" -l "$SQA_LIMIT" -o "$name/screenqa_short" ;;
        public512) echo "=== $(date +%T) $name $m: mmmu_val x$PUBLIC_LIMIT at 512 tokens"
                   step "$name" bash "$name/run_public512_phone.sh" -m "$m" -t "mmmu_val" -l "$PUBLIC_LIMIT" \
                        -o "$name/public512" ;;
        gui)       echo "=== $(date +%T) $name $m: MMBench-GUI L1 x198"
                   step "$name" env LMU_DIR="$EVAL_DIR/data/LMUData_gui198" bash "$name/run_mmbench_gui_phone.sh" -m "$m" \
                        -o "$name/mmbench_gui" ;;
        *)         echo "SKIP step $s: unknown (public, sqa, public512, gui)" ;;
      esac
    done
  done

  touch "$name/.done"; wait $wd; rm -f "$name/.done" "$name/.step_pid"
  # lmms-eval copies HF_TOKEN into *_results.json and prints it in its log; redact every file of the run
  grep -rlE 'hf_[A-Za-z0-9]{20,}' "$name" | xargs -r sed -i -E 's/hf_[A-Za-z0-9]{20,}/hf_REDACTED/g'
  tokens=$(grep -rc "hf_[A-Za-z0-9]\{20,\}" "$name" | awk -F: '{s+=$NF} END {print s+0}')
  status=done; [ -e "$name/ABORTED" ] && status="ABORTED ($(cat "$name/ABORTED"))"
  echo "=== $(date +%T) END $name $status hf_tokens=$tokens files: $(ls "$name" | tr '\n' ' ')"
done
echo ALL DONE
