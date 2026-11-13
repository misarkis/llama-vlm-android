#!/bin/bash
# Usage: [HTP_PROFILE=0|1|2] ./run_profiles.sh <folder>=<config> ...
# Runs each profile in turn with a cooldown before it, and saves each run to its own result folder
# under profiler/Results/ (override with RESULTS_DIR=<dir>).
# HTP_PROFILE (default 0) is passed to run-profile-test.ps1 as -HtpProfile; 0 is clean timing.
# Example: ./run_profiles.sh GPU-Qwen2VL-Clean=config_gpu_qwen2vl.json CPU-Qwen2VL-Clean=config_cpu_qwen2vl.json
#          HTP_PROFILE=1 ./run_profiles.sh HTP-1Core-ViT-Qwen2VL-Prof=config_htp_1core_vit_qwen2vl.json
cd "$(dirname "$0")" || exit 1
export MSYS_NO_PATHCONV=1
ADB=$(command -v adb || echo "${LOCALAPPDATA:-$HOME/AppData/Local}/Android/Sdk/platform-tools/adb.exe")
HTP=${HTP_PROFILE:-0}
RESULTS=${RESULTS_DIR:-Results}  # result folders go here, relative to profiler/
mkdir -p "$RESULTS"
DEVICE_DIR=/data/local/tmp/mtmd-profile-test

# Files run-profile-test.ps1 leaves in profiler/; all of them move into the result folder
OUTPUTS="profile_results*.csv profile_log.txt profile_log_utf8.txt thermal_report.txt
         cl_profiling.csv cl_profile_report.txt cl_trace.json
         htp_official_profile_report.txt htp_block_profile_report.txt"

zone() { "$ADB" shell "for z in /sys/class/thermal/thermal_zone*; do [ \"\$(cat \$z/type 2>/dev/null)\" = $1 ] && cat \$z/temp && break; done" | tr -d '\r'; }

cooldown() {
    for i in $(seq 1 30); do
        gpu=$(zone gpuss-0); npu=$(zone nsphvx-0); cpu=$(zone cpu-0-0-0)
        echo "$(date +%T) cooldown gpu=$gpu npu=$npu cpu=$cpu"
        if [ "${gpu:-99000}" -lt 44000 ] && [ "${npu:-99000}" -lt 44000 ] && [ "${cpu:-99000}" -lt 47000 ]; then return; fi
        command sleep 30
    done
    echo "WARNING: cooldown limit reached, phone still warm"
}

for pair in "$@"; do
    name=$RESULTS/${pair%%=*}; config=${pair#*=}
    if [ -e "$name" ]; then echo "SKIP $name: folder exists"; continue; fi
    cooldown
    mkdir "$name"
    cp "configs/$config" "$name/"
    echo "htp_profile=$HTP" > "$name/run_settings.txt"
    # Stale outputs: local ones from an earlier run, and the OpenCL files the backend writes
    # in its working directory (run-profile-test.ps1 also clears them)
    rm -f $OUTPUTS
    "$ADB" shell "rm -f $DEVICE_DIR/cl_profiling.csv $DEVICE_DIR/cl_trace.json" >/dev/null 2>&1
    echo "=== $(date +%T) START $name ($config, htp_profile=$HTP)"
    timeout 3600 pwsh -NoProfile -File run-profile-test.ps1 -Config "configs/$config" -HtpProfile "$HTP" > "$name/console.txt" 2>&1
    code=$?
    mv -f $OUTPUTS "$name/" 2>/dev/null
    log=$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['output_log'])" "$name/$config")
    "$ADB" pull "$log" "$name/" >/dev/null 2>&1
    rows=$(cat "$name"/profile_results*.csv 2>/dev/null | wc -l)
    echo "=== $(date +%T) END $name exit=$code csv_lines=$rows files: $(ls "$name" | tr '\n' ' ')"
done
echo ALL DONE
