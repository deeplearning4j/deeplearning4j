/* ******************************************************************************
*
*
* This program and the accompanying materials are made available under the
* terms of the Apache License, Version 2.0 which is available at
* https://www.apache.org/licenses/LICENSE-2.0.
*
*  See the NOTICE file distributed with this work for additional
*  information regarding copyright ownership.
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
* License for the specific language governing permissions and limitations
* under the License.
*
* SPDX-License-Identifier: Apache-2.0
******************************************************************************/

#ifndef LIBND4J_TYPES_IMPL_H
#define LIBND4J_TYPES_IMPL_H

// This header contains the cross-type constructor implementations
// that require both float16 and bfloat16 to be fully defined

#include "float16.h"
#include "bfloat16.h"

// Implementation of float16 constructor from bfloat16
SD_INLINE SD_HOST_DEVICE float16::float16(const bfloat16& value) : data{} {
  *this = static_cast<float>(value);
}

// Implementation of bfloat16 constructor from float16
SD_INLINE SD_HOST_DEVICE bfloat16::bfloat16(const float16& value) : _data(0) {
  *this = static_cast<float>(value);
}

#endif // LIBND4J_TYPES_IMPL_H
