#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
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

from torch import fx as fx
from vllm.compilation.passes.inductor_pass import get_pass_context
from vllm.compilation.passes.vllm_inductor_pass import VllmInductorPass
from vllm.config import VllmConfig

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile

"""
fusion   pass
算子融合  编译器术语: 对FX图做一遍遍历和变换

GraphFusionPassManager 就是管理一组这种变换的 pass manager
"""

class GraphFusionPassManager:
    """
    A pass manager for graph fusion passes.
    It handles the configuration and execution of passes.
    The counterpart in vllm is PostGradPassManager. Since torch_npu
    does not support triton for now, we define our own pass manager.

    一个用于图融合 pass 的 pass 管理器。
    它负责 pass 的配置和执行。
    它在 vLLM 中的对应物是 PostGradPassManager。由于 torch_npu 目前不支持 Triton，我们定义了自己的 pass 管理器。
    """

    def __init__(self):
        self.passes: list[VllmInductorPass] = []

    # 执行 pass
    def __call__(self, graph: fx.Graph) -> fx.Graph:
        compile_range = get_pass_context().compile_range

        for pass_ in self.passes:
            if pass_.is_applicable_for_range(compile_range):
                pass_(graph)
        
        # 所有 pass 跑完后 graph.recompile() 重新生成 FX 图的 Python 代码
        graph.recompile()
        return graph

    def add(self, pass_: VllmInductorPass):
        assert isinstance(pass_, VllmInductorPass)
        self.passes.append(pass_)

    # 决定启用哪些 pass
    # 还有torch.compile 自带的融合
    # npugraph_ex / torchair 后端的融合
    def configure(self, config: VllmConfig):
        from vllm_ascend.ascend_config import get_ascend_config

        # Consume the recursively validated config rather than re-reading the
        # raw additional_config dict (where e.g. "false" is truthy).

        # 读取已校验的配置
        self.ascend_compilation_config = get_ascend_config().ascend_compilation_config
        profile = get_current_hardware_profile()
        
        # 按「配置开关 + 芯片能力」
        if self.ascend_compilation_config.fuse_norm_quant and profile.supports(
            HardwareCapability.GRAPH_NORM_QUANT_FUSION
        ):
            from .passes.norm_quant_fusion_pass import AddRMSNormQuantFusionPass

            self.passes.append(AddRMSNormQuantFusionPass(config))

        if self.ascend_compilation_config.fuse_qknorm_rope:
            from .passes.qknorm_rope_fusion_pass import QKNormRopeFusionPass

            self.passes.append(QKNormRopeFusionPass(config))

        if self.ascend_compilation_config.fuse_muls_add and profile.supports(HardwareCapability.GRAPH_MULS_ADD_FUSION):
            from .passes.muls_add_pass import MulsAddFusionPass

            self.passes.append(MulsAddFusionPass(config))

        if config.compilation_config.pass_config.fuse_gemm_comms and profile.supports(
            HardwareCapability.GRAPH_MM_REDUCE_SCATTER_FUSION
        ):
            from .passes.mm_reduce_scatter_fusion_pass import MatmulReduceScatterFusionPass

            self.passes.append(MatmulReduceScatterFusionPass(config))
