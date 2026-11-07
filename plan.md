# Acceleration Plan (after ReleaseV2.0)

Baseline: `ReleaseV2.0`, llama.cpp `b12181`, NDK 27.3.13750724, Hexagon SDK 6.7.0.1, Galaxy Z Fold 7 (Adreno 830, Hexagon v79).
Replaces `plan-v2-july.md`, which was pinned to `22cadc194` and used July numbers.

Evidence tags: **[code]** read in the current source, **[log]** in the `profiler/*-Clean` results, **[old]** carried from the July plan and not re-checked.

---

## 1. Where things stand

The numbers below are the baseline. GPU numbers come from the `GPU-Only-*` runs, one run per model, which replaced the mixed GPU and NPU runs.

| Mode | Qwen2-VL decode | Qwen2-VL prefill | Qwen3.8 decode | Qwen3.8 prefill | Status |
| ---- | --------------- | ---------------- | -------------- | --------------- | ------ |
| NPU  | 40.9 t/s        | 0.72 s           | 30.9 t/s       | 0.50 s          | Valid  |
| GPU  | 29.5 t/s        | 4.62 s           | 21.8 t/s       | 3.52 s          | Valid, one run each |
| CPU  | 11.5 t/s        | 33.46 s          | 10.4 t/s       | 15.42 s         | Valid, but see check 0.2 |

Where a 130-token answer spends its time:

| Mode | Prefill share | Decode share | What to attack first |
| ---- | ------------- | ------------ | -------------------- |
| NPU, Qwen2-VL | 18% of 3.9 s | 82% | Decode speed |
| NPU, Qwen3.8  | 11% of 4.7 s | 89% | Decode speed |
| GPU, Qwen2-VL | 51% of 9.0 s | 49% | Thermal throttling |
| GPU, Qwen3.8  | 37% of 9.5 s | 63% | Thermal throttling |
| CPU, Qwen2-VL | 75% of 44.8 s | 25% | Vision encoder |
| CPU, Qwen3.8  | 55% of 27.9 s | 45% | Vision encoder |

All runs used 334x446 images, so prefill will rise at higher capture sizes. [log]

---

## 2. Problems found in the published results

**P1. GPU mode ran on GPU and NPU together. Fixed in the profiler; app fix not yet tested.** [code][log]
The profiler asked for `GPU0` and the app for `GPU`, but the device is named `GPUOpenCL`. The lookup failed, `parse_device_list` skipped it, and llama.cpp fell back to every GPU-type device, which includes the NPU. Both now ask for `GPUOpenCL`. The profiler logs show only OpenCL. The app change at `LocalInferenceService.kt:113` has not been built or run on the device.

| Run | HTP0 model buffer | OpenCL model buffer | Decode | Prefill | 130-token answer |
| --- | ----------------- | ------------------- | ------ | ------- | ---------------- |
| `GPU-Q4KM-Clean` (mixed)    | 755 MiB  | 244 MiB  | 29.7 t/s | 5.99 s | 10.4 s |
| `GPU-Only-Q4KM-Clean`       | none     | 935 MiB  | 29.5 t/s | 4.62 s | 9.0 s  |
| `GPU-Qwen38-Clean` (mixed)  | 1014 MiB | 271 MiB  | 24.7 t/s | 3.20 s | 8.5 s  |
| `GPU-Only-Qwen38-Clean`     | none     | 1205 MiB | 21.8 t/s | 3.52 s | 9.5 s  |

The mix barely changed Qwen2-VL decode, but Qwen3.8 decode is 12% slower on GPU only. Each case is one run, and the GPU-only Qwen3.8 run spent images 4 to 10 at 443 to 525 MHz, so throttling may explain the gap. Repeat it before drawing a conclusion.

Done: the README, website and `PROFILE_TEST_README.md` tables. Still to check: the in-app GPU row, if it came from GPU mode.

**P2. `image_min_tokens` does nothing.** [code]
`mtmd_params.image_min_tokens` and `image_max_tokens` are never set, and the profiler never copies its config value into `params`. The `1000` in every config has no effect, and neither does the app setting. All results ran at encoder defaults.

**P3. `vit_device: "GPU0"` is documented but fails.** [code]
`mtmd_inference_init` returns `MTMD_API_ERROR_BACKEND_INIT` for that name. Empty, `HTP0`, `CPU` and `GPUOpenCL` are the working values. Fix `PROFILE_TEST_README.md:99` and `:205`, `BUILD_LLAMACPP_ANDROID_README.md:194` and `index.html:578`.

**P4. `PromptTokens` is a position count, not a token count.** [code]
It comes from `new_n_past - old_n_past`, and an image adds `max(nx, ny)` positions. Real image tokens are never logged, so `PROFILE_TEST_README.md:221` needs rewording.

---

## 3. Status of the July findings

| ID | Finding | Status now |
| -- | ------- | ---------- |
| H1 | Image token bounds ignored | **Open** (P2) |
| H2 | `GPU` device never resolves | **Fixed** (P1) by naming `GPUOpenCL`; bad names are still skipped silently (0.1) |
| H3 | Garbage token does not stop generation | **Open**: `g_stop_generation` is set and never read, and `n_predict` can run to the limit |
| H4 | Build flags | **Open**: no KleidiAI, `GGML_OPENMP=ON` |
| H4 | `-ffp-model=fast` is not a Clang flag | **Dropped**: it is a valid Clang flag, the July claim was wrong |
| H5 | mmap forced off | **Open** on CPU and GPU, correct on NPU |
| H6 | No warmup in the app | **Open**: first image 963 ms against about 500 ms after it on NPU [log] |
| M1 | 4 threads fixed | **Open** |
| M2 | 10 ms polling, full-buffer copies | **Open** |
| M3 | No lock around the inference context | **Open** |
| M4 | Failed-run state | **Mostly fixed**: bitmaps cleared, `n_past` reset for single-turn runs |
| M5 | `llama_backend_free` on every free | **Open** |
| M6 | Per-pixel Kotlin loops | **Open** |
| M7 | Encoder and language prefill not split | **Open** |
| M8 | Encoder cost on GPU | **Superseded** for NPU mode, encoder now on HTP |
| L1 | Dead `setenv("D")` | **Open** |
| L7 | `n_ctx` 16384 | **Open**: KV is 448 MiB on Qwen2-VL [log] |
| New | Encoder on HTP in NPU mode | **Done** |
| New | One HTP session in the profiler | **Done** for the profiler; the app default still needs checking |
| New | Thinking hidden, decode speed shown | **Done** |

---

## 4. Plan

### Phase 0. Make the numbers true (do this first)

Nothing in later phases can be judged until this is done.

- 0.1 Fail or warn loudly on a bad device name instead of silently using every device. The profiler and app now name `GPUOpenCL`, but a wrong name would still fall back the same way.
- 0.2 Confirm CPU runs are CPU-only. The logs show a 12 to 40 MiB `HTP0` compute buffer in CPU mode. [log]
- 0.3 Wire `image_min_tokens` and `image_max_tokens` through `mtmd_context_params` and the profiler.
- 0.4 Log real image tokens per image, and split prefill into encoder time and language time.
- 0.5 **Done:** GPU re-run for both models as GPU-only (P1).
- 0.6 Re-run the six configs at the image sizes the app really sends, with `image_min_tokens` at a deliberate value.
- 0.7 GPU tables are corrected. Still to do: the `GPU0` doc fixes (P3), a note on the `ReleaseV2.0` release, and committing the configs that produce every published number.

Exit: every published number comes from a run whose log shows the intended device.

### Phase 1. Stop wasted time

Small changes that remove the worst user-visible delays.

- 1.1 Stop flag: check it in the decode loop, so a garbage token ends generation at once.
- 1.2 Warmup in the app at model load, so the first image is not the slow one.
- 1.3 Check the session default in the app and set it to 1 if needed. Change the `LlamaServerService.kt:195` fallback of 4.
- 1.4 Test the "ran out of tokens" message. It is still untested, the longest output so far was 344 tokens.
- 1.5 Remove `setenv("D")`.

### Phase 2. NPU decode (the largest share of the answer time)

Decode is 82 to 89% of the answer on NPU, so this is where speed comes from.

- 2.1 Tune against the FFN blocks, which took about 78% of HTP time in July [old]. Re-profile on `b12181` first.
- 2.2 Test `GGML_HEXAGON_NHVX`, VTCM size and power mode one at a time, and use `OPFILTER` to find ops falling back to CPU.
- 2.3 Test KV cache quantization (`type_k`, `type_v`). The API does not expose these yet. [code]
- 2.4 Test `n_ctx` 8192 for memory, and for speed if it changes anything.
- 2.5 Qwen3.8 with the Q8_0 encoder, already on the phone at 365 MB against 668 MB for f16.
- 2.6 Soak test of 20 or more images, since all runs so far were 10 images and NPU reached about 90°C.

Reference: the July plan cites 47 to 51 t/s upstream for this model class [old]. Check it before treating it as a target.

### Phase 3. CPU

- 3.1 Enable KleidiAI, and A/B `GGML_OPENMP` on and off.
- 3.2 Replace the fixed 4 threads with detected topology, and use a separate batch thread count.
- 3.3 Try a Q8_0 encoder, since the encoder takes 55 to 75% of the CPU answer time.
- 3.4 Turn on mmap for CPU and GPU, and measure load time and memory.

### Phase 4. GPU

GPU-only decode is 29.5 and 21.8 t/s, behind the NPU on both models, and throttling is the main limit.

- 4.1 Check whether Q4_0 helps the OpenCL path, which is still unmeasured.
- 4.2 A/B Vulkan against OpenCL.
- 4.3 Sustained thermal behaviour: the clock fell from 1200 MHz to 443 MHz. On Qwen3.8 it stayed at 443 to 525 MHz even with the GPU reading about 54°C, so find out what is holding it down.
- 4.4 Decide whether a deliberate GPU and NPU split is worth offering. It beat GPU-only on Qwen3.8 decode (24.7 against 21.8 t/s), but the NPU alone is faster than both (30.9 t/s).

### Phase 5. App pipeline

- 5.1 Replace the 10 ms polling with event callbacks (M2).
- 5.2 Add a lock around the context, so free cannot race a decode (M3).
- 5.3 Keep the backend alive for the app's lifetime, to speed up mode switches (M5).
- 5.4 Move RGB conversion to native code and remove the extra copies (M6).
- 5.5 Run the encoder while the language prefill starts, if Phase 0 shows it is worth it.

---

## 5. Priority

| Rank | Item | Why | Size of gain |
| ---- | ---- | --- | ------------ |
| 1 | Phase 0 | Image token settings, real token counts and bad device names are still open | Accuracy, no speed gain |
| 2 | Stop flag (1.1) | Removes runaway answers | Large in the worst case, not measured |
| 3 | Warmup (1.2) | First image is about 2× slower | About 0.45 s, first image only |
| 4 | NPU decode (Phase 2) | 82 to 89% of the NPU answer | Unknown until tested |
| 5 | CPU encoder and KleidiAI (Phase 3) | Slowest mode | Unknown until tested |
| 6 | GPU (Phase 4) | Slower than NPU and limited by throttling | Unknown until tested |
| 7 | App pipeline (Phase 5) | Stability, small latency gains | Tens of ms [old] |

---

## 6. Not doing

| Item | Reason |
| ---- | ------ |
| Draft-model and ngram speculative decoding | No multimodal support, and little repetition in the output [old] |
| Prompt and prefix caching | Only the system prompt repeats, while the image changes every time [old] |
| Multiple HTP sessions | They are virtual on one cDSP, and 1 session is faster than 4 |
| Vision encoder on GPU in NPU mode | HTP is faster and avoids the GPU thermal cap |
| QNN and other non-GGUF stacks | The in-tree Hexagon path already works |

---

## 7. Rules for every result

- Run at `-HtpProfile 0`, with a cooldown first, using a non-profiling build.
- Confirm the device in the log before using a number.
- Change one thing per run, and save the config with its results.
- Keep source results for every published table. The Q4_0 encoder table has no folder under `profiler/` today.
- Use the same 10 images, and state the image size next to the results.
