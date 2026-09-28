# bge-small-zh-v1.5 ONNX (small embedding model)

Dense embedding backend model for RSI hier_loop semantic retrieval
(design doc: proposal/RSI/docs/RSI-hierarchical-looped-researcher.md, module 6).

- Base model: BAAI/bge-small-zh-v1.5 (BERT, 4 layers, hidden 512, max len 512)
- ONNX conversion repo: onnx-community/bge-small-zh-v1.5-ONNX (commit 9507db33464b5da99a532ac26b2a251767cbc62b)
- Downloaded: 2026-09-28 via http://deploy.i.h.pjlab.org.cn/infra/scripts/setup_proxy.sh on login node
- Verified: SHA-256 of all onnx/ blobs matches HF LFS ETags; JSON files parse.

## Files
- onnx/model.onnx + onnx/model.onnx_data    : FP32 graph + weights (94,765,056 B)
- onnx/model_quantized.onnx + model_quantized.onnx_data : INT8 quantized (23,774,208 B)
- config.json / tokenizer.json / tokenizer_config.json

## Usage
- Set in hier config: retrieval.embedding.backend = "dense"
- Model dir: models/bge-small-zh-v1.5-onnx (model name: bge-small-zh-v1.5-onnx)
- Load with onnxruntime (CPU):
    session = onnxruntime.InferenceSession("onnx/model.onnx")  # or model_quantized.onnx
    # tokenize with tokenizer.json (BertTokenizer), mean/dump pool + l2 norm for bge
- GPU nodes are offline: copy this whole dir into any node image that runs embedding.
