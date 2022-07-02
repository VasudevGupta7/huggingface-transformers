# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

# Copyright 2022 The HuggingFace Team. All rights reserved.
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

from typing import TYPE_CHECKING

from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
)


_import_structure = {
    "configuration_data2vec": ["DATA2VEC_PRETRAINED_CONFIG_ARCHIVE_MAP", "Data2VecConfig", "Data2VecOnnxConfig"],
}

try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_flax_data2vec"] = [
        "FlaxData2VecForCausalLM",
        "FlaxData2VecForMaskedLM",
        "FlaxData2VecForMultipleChoice",
        "FlaxData2VecForQuestionAnswering",
        "FlaxData2VecForSequenceClassification",
        "FlaxData2VecForTokenClassification",
        "FlaxData2VecModel",
        "FlaxData2VecPreTrainedModel",
    ]


if TYPE_CHECKING:
    from .configuration_data2vec import DATA2VEC_PRETRAINED_CONFIG_ARCHIVE_MAP, Data2VecConfig, Data2VecOnnxConfig

    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_flax_data2vec import (
            FlaxData2VecForCausalLM,
            FlaxData2VecForMaskedLM,
            FlaxData2VecForMultipleChoice,
            FlaxData2VecForQuestionAnswering,
            FlaxData2VecForSequenceClassification,
            FlaxData2VecForTokenClassification,
            FlaxData2VecModel,
            FlaxData2VecPreTrainedModel,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
