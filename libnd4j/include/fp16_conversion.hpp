/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#ifndef CAFFE_UTIL_FP16_CONVERSION_H_
#define CAFFE_UTIL_FP16_CONVERSION_H_

#ifndef CPU_ONLY

#include <cuda_fp16.h>
// Host functions for converting between FP32 and FP16 formats
// Paulius Micikevicius (pauliusm@nvidia.com)

half cpu_float2half_rn(float f);
float cpu_half2float(half h);
half operator - (const half& h);
int isnan(const half& h);
int isinf(const half& h);

#endif // CPU_ONLY

#endif
