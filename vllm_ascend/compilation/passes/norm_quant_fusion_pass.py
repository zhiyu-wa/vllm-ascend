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
import torch
from torch._inductor.pattern_matcher import PatternMatcherPass
from vllm.compilation.passes.vllm_inductor_pass import VllmInductorPass
from vllm.config import VllmConfig
from vllm.config.compilation import Range
from vllm.logger import logger

from vllm_ascend.compilation.passes.base_pattern import BasePattern
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.utils import enable_custom_op



# 多 种“norm 后半截”的组合
# 融合 = npu_add_rms_norm_bias + 某种量化


# 无 bias + 静态 INT8 量化
class AddRMSNormQuantPattern(BasePattern):
    def __init__(self, vllm_config: VllmConfig, eps: float = 1e-6):
        super().__init__(vllm_config, eps)

    def get_inputs(self):
        """
        Generate example inputs for the AddRMSNormQuant fusion pattern.
        """
        rms_norm_input = torch.randn(2, 4, device="npu", dtype=self.dtype)
        residual = torch.randn(2, 4, device="npu", dtype=self.dtype)
        rms_norm_weight = torch.randn(4, device="npu", dtype=self.dtype)
        scale = torch.ones(4, device="npu", dtype=self.dtype)
        scale_reciprocal = torch.ones(4, device="npu", dtype=self.dtype)
        offset = torch.zeros(4, device="npu", dtype=self.dtype)
        return [rms_norm_input, residual, rms_norm_weight, scale, scale_reciprocal, offset]

    def get_pattern(self):
        def pattern(
            rms_norm_input: torch.Tensor,
            residual: torch.Tensor,
            rms_norm_weight: torch.Tensor,
            scale: torch.Tensor,
            scale_reciprocal: torch.Tensor,
            offset: torch.Tensor,
        ):
            """
            Pattern for AddRMSNormQuant fusion.
            """
            # 无 bias 版：RMSNorm 的加性偏置
            output = torch.ops._C_ascend.npu_add_rms_norm_bias(
                rms_norm_input, residual, rms_norm_weight, None, self.eps
            )
            out0 = output[0]
            out1 = output[2]
            # 自定义算子
            quantized_output = torch.ops.vllm.quantize(out0, scale, scale_reciprocal, offset)
            return quantized_output, out1

        return pattern

    def get_replacement(self):
        def replacement(
            rms_norm_input: torch.Tensor,
            residual: torch.Tensor,
            rms_norm_weight: torch.Tensor,
            scale: torch.Tensor,
            scale_reciprocal: torch.Tensor,
            offset: torch.Tensor,
        ):
            """
            Replacement for the AddRMSNormQuant fusion.
            """
            # 残差 + RMS归一化 + 量化
            output = torch.ops.npu.npu_add_rms_norm_quant(
                rms_norm_input,             # hidden states
                residual,                   # 残差
                rms_norm_weight,            # rms_norm的权重 weight 
                scale,                      # 量化的scale 放大倍数，把浮点世界的小数，换算成整数世界的"格数"
                offset,                     # 量化的零点，平移量
                epsilon=self.eps            # 防止除以 0
            )
            
            quantized_output = output[0]    # 量化后的 norm 结果
            out1 = output[2]                # 更新后的残差
            return quantized_output, out1

        return replacement


# 有 bias + 静态 INT8
class AddRMSNormQuantPatternWithBias(BasePattern):
    def __init__(self, vllm_config: VllmConfig, eps: float = 1e-6):
        super().__init__(vllm_config, eps)

    def get_inputs(self):
        """
        Generate example inputs for the AddRMSNormQuant fusion pattern.
        """
        rms_norm_input = torch.randn(2, 4, device="npu", dtype=self.dtype)
        residual = torch.randn(2, 4, device="npu", dtype=self.dtype)
        rms_norm_weight = torch.randn(4, device="npu", dtype=self.dtype)
        rmsnorm_bias = torch.randn(4, device="npu", dtype=self.dtype)
        scale = torch.ones(4, device="npu", dtype=self.dtype)
        scale_reciprocal = torch.ones(4, device="npu", dtype=self.dtype)
        offset = torch.zeros(4, device="npu", dtype=self.dtype)
        return [rms_norm_input, residual, rms_norm_weight, scale, scale_reciprocal, offset, rmsnorm_bias]

    def get_pattern(self):
        def pattern(
            rms_norm_input: torch.Tensor,
            residual: torch.Tensor,
            rms_norm_weight: torch.Tensor,
            scale: torch.Tensor,
            scale_reciprocal: torch.Tensor,
            offset: torch.Tensor,
            bias: torch.Tensor,
        ):
            """
            Pattern for AddRMSNormQuant fusion.
            """
            output = torch.ops._C_ascend.npu_add_rms_norm_bias(
                rms_norm_input, residual, rms_norm_weight, bias, self.eps
            )
            out0 = output[0]
            out1 = output[2]
            quantized_output = torch.ops.vllm.quantize(out0, scale, scale_reciprocal, offset)
            return quantized_output, out1

        return pattern

    def get_replacement(self):
        def replacement(
            rms_norm_input: torch.Tensor,
            residual: torch.Tensor,
            rms_norm_weight: torch.Tensor,
            scale: torch.Tensor,
            scale_reciprocal: torch.Tensor,
            offset: torch.Tensor,
            bias: torch.Tensor,
        ):
            """
            Replacement for the AddRMSNormQuant fusion.
            """
            output = torch.ops.npu.npu_add_rms_norm_quant(
                rms_norm_input, residual, rms_norm_weight, scale, offset, epsilon=self.eps, beta=bias
            )
            quantized_output = output[0]
            out1 = output[2]
            return quantized_output, out1

        return replacement


# 无 bias + 动态 INT8
class AddRMSNormDynamicQuantPattern(BasePattern):
    def __init__(self, vllm_config: VllmConfig, eps: float = 1e-6):
        super().__init__(vllm_config, eps)

    def get_inputs(self):
        """
        Generate example inputs for the AddRMSNormQuant fusion pattern.
        """
        rms_norm_input = torch.randn(2, 4, device="npu", dtype=self.dtype)
        residual = torch.randn(2, 4, device="npu", dtype=self.dtype)
        rms_norm_weight = torch.randn(4, device="npu", dtype=self.dtype)
        return [rms_norm_input, residual, rms_norm_weight]

    def get_pattern(self):
        def pattern(rms_norm_input: torch.Tensor, residual: torch.Tensor, rms_norm_weight: torch.Tensor):
            """
            Pattern for AddRMSNormQuant fusion.
            qwen3.8-dense 
            """
            output = torch.ops.npu.npu_add_rms_norm(rms_norm_input, residual, rms_norm_weight, self.eps)
            out0 = output[0]
            out1 = output[2]
            quantized_output = torch.ops.npu.npu_dynamic_quant(out0)
            return quantized_output[0], quantized_output[1], out1

        return pattern

    def get_replacement(self):
        def replacement(rms_norm_input: torch.Tensor, residual: torch.Tensor, rms_norm_weight: torch.Tensor):
            """
            Replacement for the AddRMSNormQuant fusion.
            """
            output = torch.ops.npu.npu_add_rms_norm_dynamic_quant(
                rms_norm_input, residual, rms_norm_weight, epsilon=self.eps, output_mask=[True, False]
            )
            return (
                output[0],
                output[3],
                output[2],
            )

        return replacement


# 有 bias + 动态 INT8
class AddRMSNormDynamicQuantPatternWithBias(BasePattern):
    def __init__(self, vllm_config: VllmConfig, eps: float = 1e-6):
        super().__init__(vllm_config, eps)

    def get_inputs(self):
        """
        Generate example inputs for the AddRMSNormQuant fusion pattern.
        """
        rms_norm_input = torch.randn(2, 4, device="npu", dtype=self.dtype)
        residual = torch.randn(2, 4, device="npu", dtype=self.dtype)
        rms_norm_weight = torch.randn(4, device="npu", dtype=self.dtype)
        rmsnorm_bias = torch.randn(4, device="npu", dtype=self.dtype)
        return [rms_norm_input, residual, rms_norm_weight, rmsnorm_bias]

    def get_pattern(self):
        def pattern(
            rms_norm_input: torch.Tensor,
            residual: torch.Tensor,
            rms_norm_weight: torch.Tensor,
            bias: torch.Tensor,
        ):
            """
            Pattern for AddRMSNormQuant fusion.
            """
            output = torch.ops._C_ascend.npu_add_rms_norm_bias(
                rms_norm_input, residual, rms_norm_weight, bias, self.eps
            )
            out0 = output[0]
            out1 = output[2]
            quantized_output = torch.ops.npu.npu_dynamic_quant(out0)
            return quantized_output[0], quantized_output[1], out1

        return pattern

    def get_replacement(self):
        def replacement(
            rms_norm_input: torch.Tensor,
            residual: torch.Tensor,
            rms_norm_weight: torch.Tensor,
            bias: torch.Tensor,
        ):
            """
            Replacement for the AddRMSNormQuant fusion.
            """
            # 遇到上面这个 op 序列就替换成下面的融合算子
            output = torch.ops.npu.npu_add_rms_norm_dynamic_quant(
                rms_norm_input, residual, rms_norm_weight, epsilon=self.eps, output_mask=[True, False], beta=bias
            )
            return (
                output[0],
                output[3],
                output[2],
            )

        return replacement


class AddRMSNormDynamicMXQuantPattern(BasePattern):
    def __init__(self, vllm_config: VllmConfig, eps: float = 1e-6):
        super().__init__(vllm_config, eps)

    def get_inputs(self):
        """
        Generate example inputs for the AddRMSNormDynamicMXQuant fusion pattern.
        """
        rms_norm_input = torch.randn(2, 64, device="npu", dtype=self.dtype)
        residual = torch.randn(2, 64, device="npu", dtype=self.dtype)
        rms_norm_weight = torch.randn(64, device="npu", dtype=self.dtype)
        return [rms_norm_input, residual, rms_norm_weight]

    def get_pattern(self):
        def pattern(rms_norm_input: torch.Tensor, residual: torch.Tensor, rms_norm_weight: torch.Tensor):
            """
            Pattern for AddRMSNormDynamicMXQuant fusion.
            """
            output = torch.ops.npu.npu_add_rms_norm(rms_norm_input, residual, rms_norm_weight, self.eps)
            out0 = output[0]
            out1 = output[2]
            quantized_output = torch.ops.npu.npu_dynamic_mx_quant(out0, dst_type=torch.float8_e4m3fn)
            return quantized_output[0], quantized_output[1], out1

        return pattern

    def get_replacement(self):
        def replacement(rms_norm_input: torch.Tensor, residual: torch.Tensor, rms_norm_weight: torch.Tensor):
            """
            Replacement for the AddRMSNormDynamicMXQuant fusion.
            """
            output = torch.ops.npu.npu_add_rms_norm_dynamic_mx_quant(
                rms_norm_input,
                residual,
                rms_norm_weight,
                epsilon=self.eps,
                dst_type=torch.float8_e4m3fn,
            )
            return (
                output[0],
                output[2],
                output[1],
            )

        return replacement


class RMSNormDynamicMXQuantPattern(BasePattern):
    def __init__(self, vllm_config: VllmConfig, eps: float = 1e-6):
        super().__init__(vllm_config, eps)

    def get_inputs(self):
        """
        Generate example inputs for the RMSNormDynamicMXQuant fusion pattern.
        """
        rms_norm_input = torch.randn(2, 64, device="npu", dtype=self.dtype)
        rms_norm_weight = torch.randn(64, device="npu", dtype=self.dtype)
        return [rms_norm_input, rms_norm_weight]

    def get_pattern(self):
        def pattern(rms_norm_input: torch.Tensor, rms_norm_weight: torch.Tensor):
            """
            Pattern for RMSNormDynamicMXQuant fusion.
            """
            output = torch.ops.npu.npu_rms_norm(rms_norm_input, rms_norm_weight, self.eps)
            out0 = output[0]
            quantized_output = torch.ops.npu.npu_dynamic_mx_quant(out0, dst_type=torch.float8_e4m3fn)
            return quantized_output[0], quantized_output[1]

        return pattern

    def get_replacement(self):
        def replacement(rms_norm_input: torch.Tensor, rms_norm_weight: torch.Tensor):
            """
            Replacement for the RMSNormDynamicMXQuant fusion.
            """
            output = torch.ops.npu.npu_rms_norm_dynamic_mx_quant(
                rms_norm_input,
                rms_norm_weight,
                epsilon=self.eps,
                dst_type=torch.float8_e4m3fn,
            )
            return output[0], output[1]

        return replacement


def _model_uses_w4a4_quant(vllm_config: VllmConfig | None) -> bool:
    """Check whether the model uses W4A4 int4 quantization for any layer.

    W4A4 int4 schemes (e.g. W4A4_DYNAMIC, W4A4_FLATQUANT_DYNAMIC)
    are incompatible with the fuse_norm_quant optimization,
    so callers use this to disable that fusion.
    """
    if vllm_config is None:
        return False
    quant_config = getattr(vllm_config, "quant_config", None)
    if quant_config is None:
        return False
    quant_description = getattr(quant_config, "quant_description", None)
    if not quant_description:
        return False
    w4a4_int4_schemes = ["W4A4_DYNAMIC", "W4A4_FLATQUANT_DYNAMIC"]
    return any(
        isinstance(quant_type, str) and quant_type in w4a4_int4_schemes for quant_type in quant_description.values()
    )


class AddRMSNormQuantFusionPass(VllmInductorPass):
    """
    A pass for fusing AddRMSNorm and W8A8 quantization operations on Ascend.
    """

    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)
        # 每个 pass 实例一个 matcher
        self.pattern_match_passes: PatternMatcherPass = PatternMatcherPass(pass_name="rmsnorm_quant_fusion_pass")

        # 模型 dtype 必须是 bf16/fp16
        dtype = vllm_config.model_config.dtype
        if dtype not in (torch.bfloat16, torch.float16):
            logger.debug("Quant fusion not enabled: unsupported dtype %s", dtype)
            # matcher 保持空 pass 注册了但不会替换任何东西
            return

        # W4A4 模型直接 return
        if _model_uses_w4a4_quant(vllm_config):
            logger.debug(
                "Quant fusion not enabled: the model contains "
                "W4A4 quantized weights, which are incompatible with the "
                "norm-quant fusion pass."
            )
            return

        # 同一套 pattern 按两个 eps 各注册一遍，因为 FX 图里 eps 是常量，不同值要分别匹配
        common_epsilons = [1e-5, 1e-6]
        profile = get_current_hardware_profile()

        for eps in common_epsilons:
            
            # 无条件注册 AddRMSNormDynamicQuantPattern
            # 最基础的 dynamic INT8 融合
            AddRMSNormDynamicQuantPattern(vllm_config, eps=eps).register(self.pattern_match_passes)
            
            # 只有硬件支持 DYNAMIC_MX_QUANT_FUSION 
            # 才注册两条 MX（FP8）路径
            if profile.supports(HardwareCapability.DYNAMIC_MX_QUANT_FUSION):
                AddRMSNormDynamicMXQuantPattern(vllm_config, eps=eps).register(self.pattern_match_passes)
                RMSNormDynamicMXQuantPattern(vllm_config, eps=eps).register(self.pattern_match_passes)
            
            # 只有 enable_custom_op() 当前环境能不能安全使用自定义算子库 为真，
            # 才注册四条 static/带 bias 的 INT8 pattern
            if enable_custom_op():
                AddRMSNormQuantPattern(vllm_config, eps=eps).register(self.pattern_match_passes)
                AddRMSNormQuantPatternWithBias(vllm_config, eps=eps).register(self.pattern_match_passes)
                AddRMSNormDynamicQuantPatternWithBias(vllm_config, eps=eps).register(self.pattern_match_passes)

    def __call__(self, graph: torch.fx.Graph):
        # 传入FX图
        self.begin()
        # 统计匹配数量
        self.matched_count = self.pattern_match_passes.apply(graph)
        # 打印
        logger.debug("Replaced %s patterns", self.matched_count)
        self.end_and_log()

    # 这个 pass 对 prefill / decode / speculator 所有 compile range 都执行
    def is_applicable_for_range(self, compile_range: Range) -> bool:
        """
        Check if the pass is applicable for the current configuration.
        """
        return True
