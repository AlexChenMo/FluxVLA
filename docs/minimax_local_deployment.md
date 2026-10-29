# Self-hosted MiniMax-M3.1-Flash

The MiniMax policy speaks the OpenAI-compatible Chat Completions protocol, so
a self-hosted [MiniMax-M3.1-Flash](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash)
served by vLLM or SGLang can replace the hosted API without changing the
policy code. Only the endpoint changes; the observations, the tool call and
the action conversion are identical on both paths.

The checkpoint is loaded by the inference server, not by FluxVLA. FluxVLA
sends the same request either way.

## Requirements

- The checkpoint is a 428B mixture-of-experts model with 23B active
  parameters and about 252 GB of weights. Download it once:

```bash
hf download MiniMaxAI/MiniMax-M3.1-Flash --local-dir MiniMax-M3.1-Flash
```

- vLLM 0.33.0 or newer. The model is not yet in the released PyPI wheel, so
  use the nightly build or the prebuilt image:

```bash
pip install vllm --pre
# or
docker pull vllm/vllm-openai:minimax-m31-flash
```

- Eight H100 or H200 GPUs for a single node. The recipe below is the
  official single-node configuration for Hopper.

## Serving

```bash
vllm serve MiniMaxAI/MiniMax-M3.1-Flash \
  --tensor-parallel-size 8 \
  --reasoning-parser minimax_m3 \
  --block-size 128 \
  --kv-cache-dtype fp8 \
  --attention-config '{"indexer_kv_dtype": "fp8"}' \
  --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
  --max-model-len 262144 \
  --max-num-seqs 80 \
  --limit-mm-per-prompt '{"video": 0}' \
  --gpu-memory-utilization 0.9 \
  --no-enable-log-requests \
  --disable-hybrid-kv-cache-manager \
  --moe-backend marlin \
  --enable-auto-tool-choice \
  --tool-call-parser minimax_m3
```

Three flags are mandatory for this policy:

- `--enable-auto-tool-choice` with `--tool-call-parser minimax_m3` makes the
  server return `move_to` / `control_gr1` as a structured tool call. Without
  them the model still emits the call as text and the policy cannot parse it.
- `--block-size 128` is required by the MSA sparse attention indexer.
- `--limit-mm-per-prompt '{"video": 0}'` disables video inputs. Image inputs
  stay enabled.

Hopper has no FP4 tensor cores, so the KV cache and the indexer K cache use
FP8 and the NVFP4 experts run on the Marlin MoE kernel. On Blackwell the
NVFP4 variants apply instead.

## Evaluation

Point the config at the local server and run the normal evaluation command:

```bash
export MINIMAX_LOCAL_BASE_URL='http://127.0.0.1:8000/v1'
export MINIMAX_LOCAL_SERVED_MODEL_NAME='MiniMax-M3.1-Flash'

bash scripts/eval.sh \
  configs/minimax/minimax_m31_flash_local_libero_10_inference.py \
  --cfg-options eval.task_ids=0 eval.num_trials_per_task=1 \
  inference_model.max_llm_calls=5 eval.max_steps=60
```

The RoboCasa GR1 suite uses
`configs/minimax/minimax_m31_flash_local_robocasa_inference.py`.

Available configurations:

| Config                                           | Suite                   |
| ------------------------------------------------ | ----------------------- |
| `minimax_m31_flash_local_libero_inference.py`    | LIBERO-Object           |
| `minimax_m31_flash_local_libero_10_inference.py` | LIBERO-10               |
| `minimax_m31_flash_local_robocasa_inference.py`  | RoboCasa GR1 (24 tasks) |

## Environment

| Variable                          | Purpose                                                         |
| --------------------------------- | --------------------------------------------------------------- |
| `MINIMAX_LOCAL_BASE_URL`          | Server base URL, defaults to `http://127.0.0.1:8000/v1`         |
| `MINIMAX_LOCAL_SERVED_MODEL_NAME` | Model name sent to the server, defaults to `MiniMax-M3.1-Flash` |

The hosted `MINIMAX_CN_API_KEY` and `MINIMAX_INTL_API_KEY` variables are not
used by the local configs. The policy never reads an API key in this mode, so
a hosted key can never be sent to a self-hosted endpoint.

### The served model name

A server registers the model under the name it was started with, and rejects
requests for any other name with HTTP 404. `vllm serve` defaults
`--served-model-name` to the model path, which is often a long local path,
so set `MINIMAX_LOCAL_SERVED_MODEL_NAME` to whatever the server was started
with, or pass `--served-model-name` when starting the server.

### Proxies

A self-hosted server listens on loopback, but an `http_proxy` in the
environment would be used for it and fail with a proxy error. The policy
bypasses the proxy for loopback hosts only, so the hosted endpoints still work
behind a corporate proxy. If you export the proxy variables yourself, exclude
the server address:

```bash
export no_proxy="localhost,127.0.0.1,::1"
export NO_PROXY="localhost,127.0.0.1,::1"
```

## Notes

- The local server ignores the hosted API key, so `allow_anonymous=True` is
  set in the local configs. In this mode the policy does not read an API key
  at all, so no credential can reach a self-hosted endpoint. It never changes
  the request body.
- `explicit_tool_choice=True` is set in the local configs. OpenAI-compatible
  servers default to `tool_choice='none'`, which forbids tool calls, so the
  field has to be sent explicitly. MiniMax's own Chat Completions reference
  does not document `tool_choice` or `parallel_tool_calls`, so the hosted
  configs keep omitting them.
- The policy requests `reasoning_effort=low`. The server maps it to the chat
  template, and the model always thinks, so thinking tokens count towards
  `max_output_tokens` (16384 by default).
- The chat template expects the complete assistant turn, including the
  thinking block, to be sent back during multi-turn tool use. The policy
  replays the raw assistant turn verbatim.
- `image_url.detail` is not interpreted by the server, but it is accepted and
  does not affect the request.
- A self-hosted server answers more slowly than the hosted API. The local
  configs raise `request_timeout` to 300 seconds.
