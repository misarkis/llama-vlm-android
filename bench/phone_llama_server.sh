#!/bin/bash
# Copyright (c) 2026 Michel Sarkis (https://github.com/misarkis)
#
# This software is released under the MIT License.
# https://opensource.org

# Stand-in for llama-server that runs llama-server on the phone's NPU, behind `adb forward`.
#
# eval/run_public.sh and eval/run_screenqa_short.sh start
#   "$LLAMA_SERVER" <model args> -c CTX -np SLOTS -ngl 99 --jinja ... --host 127.0.0.1 --port PORT
# and stop it with kill. Pointing LLAMA_SERVER at this script runs the same llama-server build on
# the phone instead, so the harnesses and their official scorers run unchanged.
#
#   bench/phone_llama_server.sh --setup     push llama-server + libs to the phone (once per build)
#   (normally started by bench/run_evals.sh)
#
# What it changes from the PC command line:
#   - model paths under $MODEL_DIR/ map to $PHONE_MODELS/ (same folder and file names)
#   - -c becomes $PHONE_CTX (the device's profiler setup: Flip6 8192, Fold 7 16384), -np becomes 1
#   - adds --device $PHONE_DEVICE (default HTP0, the NPU-1Core setup of the profiler)
#   - waits for the phone to cool down before starting (profiler thresholds)
set -u
export MSYS_NO_PATHCONV=1
REPO="$(cd "$(dirname "$0")/.." && pwd)"
# Settings from .env (written by setup-config.sh / setup-config.ps1); a value the caller already set wins
ENV_FILE="$REPO/.env"
if [ -f "$ENV_FILE" ]; then
  while IFS='=' read -r k v || [ -n "$k" ]; do
    v="${v%$'\r'}"
    [[ $k =~ ^[A-Z_][A-Z0-9_]*$ ]] && [ -z "${!k:-}" ] && export "$k=$v"
  done < "$ENV_FILE"
fi
need() { for k; do [ -n "${!k:-}" ] || { echo "$k is not set: run setup-config.sh (it writes .env)"; exit 1; }; done; }
need ANDROID_SDK_ROOT ANDROID_NDK_HOME MODEL_DIR
ADB="$ANDROID_SDK_ROOT/platform-tools/adb"
BUILD="$REPO/llama.cpp/build-android"
DEV_DIR=/data/local/tmp/llama-server-eval
PHONE_MODELS="${PHONE_MODELS:-/data/local/tmp/llama.cpp/models}"
PHONE_CTX="${PHONE_CTX:?set PHONE_CTX (Flip6 8192, Fold 7 16384)}"
PHONE_DEVICE="${PHONE_DEVICE:-HTP0}"
PHONE_EXTRA="${PHONE_EXTRA:-}"   # extra server flags from the config (--cache-ram 0: the prompt cache filled the Flip6 RAM
                                 # and rebooted it; --no-host at 8192 ctx was checked and works)

if [ "${1:-}" = "--setup" ]; then
  "$ADB" shell "mkdir -p $DEV_DIR"
  for f in bin/llama-server bin/libllama-server-impl.so bin/libllama.so bin/libmtmd.so bin/libggml.so \
           bin/libggml-base.so bin/libggml-cpu.so bin/libggml-opencl.so bin/libggml-hexagon.so \
           bin/libllama-common.so ggml/src/ggml-hexagon/libggml-htp-v73.so \
           ggml/src/ggml-hexagon/libggml-htp-v75.so ggml/src/ggml-hexagon/libggml-htp-v79.so \
           ggml/src/ggml-hexagon/libggml-htp-v81.so; do
    "$ADB" push "$(cygpath -w "$BUILD/$f")" "$DEV_DIR/" >/dev/null || echo "push failed: $f"
  done
  # OpenMP runtime from the NDK (libggml-base.so needs it), as the profiler does
  ndk="$ANDROID_NDK_HOME"
  omp=$(ls "$ndk"/toolchains/llvm/prebuilt/windows-x86_64/lib*/clang/*/lib/linux/aarch64/libomp.so 2>/dev/null | head -1)
  if [ -n "$omp" ]; then "$ADB" push "$(cygpath -w "$omp")" "$DEV_DIR/" >/dev/null; else echo "libomp.so not found under $ndk"; fi
  "$ADB" shell "chmod +x $DEV_DIR/llama-server"
  exit 0
fi

# Rewrite the PC arguments for the phone
args=(); port=8081
while [ $# -gt 0 ]; do
  case "$1" in
    -c)     args+=(-c "$PHONE_CTX"); shift 2 ;;
    -np)    args+=(-np 1); shift 2 ;;
    --host) shift 2 ;;
    --port) port="$2"; shift 2 ;;
    *)      a="$1"; [[ "$a" == "$MODEL_DIR"/* ]] && a="$PHONE_MODELS/${a#"$MODEL_DIR"/}"
            args+=("$a"); shift ;;
  esac
done
# shellcheck disable=SC2206
args+=(--device "$PHONE_DEVICE" $PHONE_EXTRA --host 127.0.0.1 --port "$port")

# Quote each argument for the phone's shell (the chat-template JSON needs it)
remote=""; for a in "${args[@]}"; do remote+=" '${a//\'/\'\\\'\'}'"; done

zone() { "$ADB" shell "for z in /sys/class/thermal/thermal_zone*; do [ \"\$(cat \$z/type 2>/dev/null)\" = $1 ] && cat \$z/temp && break; done" | tr -d '\r'; }
for _ in $(seq 1 30); do
  npu=$(zone nsphvx-0); cpu=$(zone cpu-0-0-0)
  echo "$(date +%T) cooldown npu=$npu cpu=$cpu"
  [ "${npu:-99000}" -lt 44000 ] && [ "${cpu:-99000}" -lt 47000 ] && break
  sleep 30
done

cleanup() {
  "$ADB" shell "pkill -f $DEV_DIR/llama-server" >/dev/null 2>&1
  "$ADB" forward --remove "tcp:$port" >/dev/null 2>&1
  exit 0
}
trap cleanup EXIT INT TERM

echo "phone llama-server:$remote"
"$ADB" forward "tcp:$port" "tcp:$port" >/dev/null
"$ADB" shell "cd $DEV_DIR && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. exec $DEV_DIR/llama-server$remote" &
wait $!
