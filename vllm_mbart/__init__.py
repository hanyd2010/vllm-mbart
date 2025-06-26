# SPDX-License-Identifier: Apache-2.0

from vllm import ModelRegistry


def register():
    from .mbart import MBartForConditionalGeneration

    # "BartModel": ("bart", "BartForConditionalGeneration"),
    # "BartForConditionalGeneration": ("bart", "BartForConditionalGeneration"),
    if "MBartModel" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model("MBartModel", MBartForConditionalGeneration)

    if "MBartForConditionalGeneration" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model("MBartForConditionalGeneration", MBartForConditionalGeneration)
