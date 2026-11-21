#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#
"""Lightweight input contract shared by quantization and MoE modules."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PreparedLinearInput:
    # 在 multistream scheduling barrier 多流调度 之前准备好的量化输入
    # NPU 上有多个硬件流（stream）可以进行并行
    # 
    """Quantized input prepared before a multistream scheduling barrier."""

    quantized: torch.Tensor
    scale: torch.Tensor | None
    output_dtype: torch.dtype


"""
barrier 是什么 （屏障）
它不是一种算子，是一种 "等待"

Event_wait
Event_record

共享专家流在 barrier 前把做量化（和路由流的 Cube 工作重叠），量化产物打包成这个对象穿过 event 等待，barrier 后 apply 直接 GEMM

"""