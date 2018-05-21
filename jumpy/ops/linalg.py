# Copyright 2016 Skymind,Inc. All Rights Reserved.
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
# ==============================================================================

from .op import op
from ..java_classes import *


# Linear algebra
# https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.linalg.html


@op
def dot(arr, other):
	return arr.mmul(other)


@op
def tensordot(arr1, arr2, axes=2):
	shape1 = arr1.shape()
	shape2 = arr2.shape()
	if type(axes) is int:
		axes = [shape1[axes:], shape2[:axes]]
	elif type(axes) in [list, tuple]:
		axes = list(axes)
		for i in range(2):
			if type(axes[i]) is int:
				axes[i] = [axes[i]]
	return Nd4j.tensorMmul(arr1, arr2, axes)
