# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Checkpoint-free MiniMax-M3.1-Flash evaluation on LIBERO, self-hosted.

This config drives a locally served MiniMax-M3.1-Flash instead of the hosted
API. The policy speaks the OpenAI-compatible Chat Completions protocol, so a
local inference server can serve the checkpoint unchanged::

    vllm serve MiniMaxAI/MiniMax-M3.1-Flash \\
      --tensor-parallel-size 8 \\
      --reasoning-parser minimax_m3 \\
      --block-size 128 \\
      --kv-cache-dtype fp8 \\
      --attention-config '{"indexer_kv_dtype": "fp8"}' \\
      --max-model-len 262144 \\
      --limit-mm-per-prompt '{"video": 0}' \\
      --enable-auto-tool-choice \\
      --tool-call-parser minimax_m3

``--enable-auto-tool-choice`` with ``--tool-call-parser minimax_m3`` is
required so the ``move_to`` call is returned as a structured tool call, and
``--block-size 128`` is required by the MSA sparse attention indexer.

Connection settings are read from the environment::

    export MINIMAX_LOCAL_BASE_URL='http://127.0.0.1:8000/v1'
    export MINIMAX_LOCAL_SERVED_MODEL_NAME='MiniMax-M3.1-Flash'

The served model name must match the server's ``--served-model-name``, which
defaults to the model path. The hosted ``MINIMAX_CN_API_KEY`` and
``MINIMAX_INTL_API_KEY`` variables are not used here: a local server does not
authenticate requests, so the policy sends no ``Authorization`` header and
never reads an API key.
"""

_environ = __import__('os').environ

_MINIMAX_LOCAL_BASE_URL = (
    _environ.get('MINIMAX_LOCAL_BASE_URL') or 'http://127.0.0.1:8000/v1')
_MINIMAX_LOCAL_SERVED_MODEL_NAME = (
    _environ.get('MINIMAX_LOCAL_SERVED_MODEL_NAME') or 'MiniMax-M3.1-Flash')

inference_model = dict(
    type='MiniMaxLiberoVLA',
    model='MiniMax-M3.1-Flash',
    # A local server speaks the OpenAI-compatible protocol, so the hosted
    # Anthropic-compatible Messages API is not used.
    api_protocol='chat_completions',
    base_url=_MINIMAX_LOCAL_BASE_URL,
    anthropic_base_url=_MINIMAX_LOCAL_BASE_URL,
    served_model_name=_MINIMAX_LOCAL_SERVED_MODEL_NAME,
    # The local server does not require authentication. This also stops the
    # hosted API key from being read, so it can never leak to a self-hosted
    # endpoint.
    allow_anonymous=True,
    # OpenAI-compatible servers default to tool_choice='none', which forbids
    # tool calls, so the field has to be sent explicitly.
    explicit_tool_choice=True,
    tool_choice='auto',
    # M3.1-Flash always thinks; low effort bounds closed-loop latency.
    reasoning_effort='low',
    # Thinking tokens count towards this cap, so keep ample headroom.
    max_output_tokens=16384,
    # A self-hosted server answers more slowly than the hosted API.
    request_timeout=300.0,
    max_retries=2,
    retry_backoff=2.0,
    image_detail='default',
    image_format='JPEG',
    jpeg_quality=95,
    image_horizon=2,
    max_llm_calls=50,
    # One API move is converted to at most ten LIBERO control steps.
    action_horizon=10,
    max_speed_fraction=0.5,
    position_action_scale=0.01,
    rotation_action_scale=0.1,
    max_rotation_speed_fraction=0.25,
    gripper_settle_steps=8,
    workspace_bounds=[[-0.45, 0.45], [-0.45, 0.45], [-0.05, 1.40]],
    task_visual_hints={
        'pick up the alphabet soup and place it in the basket':
        ('The alphabet soup is the blue-and-orange cylindrical can with '
         'large yellow ALPHABET SOUP lettering. It is not the red-and-green '
         'tomato sauce can.'),
    },
)

# Keep environment helpers out of MMEngine's serialized config namespace.
del _environ, _MINIMAX_LOCAL_BASE_URL, _MINIMAX_LOCAL_SERVED_MODEL_NAME

eval = dict(
    type='LiberoEvalRunner',
    task_suite_name='libero_object',
    task_ids=None,
    model_family='minimax-m3.1-flash-local',
    eval_chunk_size=10,
    resize_size=512,
    num_trials_per_task=1,
    num_steps_wait=10,
    # Fifty ten-step API action chunks plus settling/reset headroom.
    max_steps=550,
    seed=7,
    norm_stats_path=None,
    requires_dataset_stats=False,
    dataset=dict(
        type='OpenAILiberoEvalDataset',
        img_keys=['agentview_image', 'robot0_eye_in_hand_image'],
        resize_size=512,
    ),
    denormalize_action=dict(type='IdentityLiberoAction', action_dim=7),
    model_build_device='cpu',
    enable_mixed_precision_training=False,
    preprocess_every_step=False,
    save_rollout_videos=True,
    save_failed_rollout_videos=False,
    save_multi_view_rollout_videos=True,
    output_dir='work_dirs/minimax_m31_flash_local_libero',
)
