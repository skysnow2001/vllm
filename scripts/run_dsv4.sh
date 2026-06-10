VLLM_DSV4_TRITON=1 \
  MOE_BACKEND=triton_unfused \
  CUDAGRAPH_MODE=FULL_AND_PIECEWISE \
  MODEL_PATH=/app/models/deepseek-ai/DeepSeek-V4-Flash \
  scripts/serve_deepseek_v4_flash.sh