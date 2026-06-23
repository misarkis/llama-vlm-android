# Assets Directory

Place model files and configuration here:

## Required Files

1. **config.json** - VLM configuration (already included)
2. **qwen3.5.gguf** - VLM model file (large, download separately)
3. **sherpa-onnx-whisper-base.en/** - Whisper models for voice recognition
   - Downloaded automatically by `build-sherpa-from-source.sh`
   - Contains: `base.en-encoder.int8.onnx`, `base.en-decoder.int8.onnx`, `base.en-tokens.txt`
4. **vits-model.onnx** - TTS model (if using local VITS TTS)

## Note

The config.json is included as a template. The actual VLM API calls go to a remote llama.cpp server, so the large model files are not needed for the basic screen analysis feature.

For local inference with the Qwen model, download the GGUF file and place it here.

## Voice Recognition

Voice input uses sherpa-onnx with Whisper base.en model. The model files are automatically downloaded and placed in assets by the build script:

```bash
./build-sherpa-from-source.sh v1.13.2
```
