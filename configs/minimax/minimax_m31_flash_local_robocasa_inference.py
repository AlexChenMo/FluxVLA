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
"""Checkpoint-free MiniMax-M3.1-Flash evaluation on RoboCasa GR1, self-hosted.

Serves the checkpoint locally instead of calling the hosted API; see
``configs/minimax/minimax_m31_flash_local_libero_inference.py`` for the server
requirements and the environment variables.
"""

_environ = __import__('os').environ

_MINIMAX_LOCAL_BASE_URL = (
    _environ.get('MINIMAX_LOCAL_BASE_URL') or 'http://127.0.0.1:8000/v1')
_MINIMAX_LOCAL_SERVED_MODEL_NAME = (
    _environ.get('MINIMAX_LOCAL_SERVED_MODEL_NAME') or 'MiniMax-M3.1-Flash')

_base_ = './minimax_m31_flash_robocasa_inference.py'

inference_model = dict(
    # A local server speaks the OpenAI-compatible protocol.
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
    # A self-hosted server answers more slowly than the hosted API.
    request_timeout=300.0,
)

# Keep environment helpers out of MMEngine's serialized config namespace.
del _environ, _MINIMAX_LOCAL_BASE_URL, _MINIMAX_LOCAL_SERVED_MODEL_NAME

eval = dict(
    model_family='minimax-m3.1-flash-local',
    output_dir='work_dirs/minimax_m31_flash_local_robocasa',
)
