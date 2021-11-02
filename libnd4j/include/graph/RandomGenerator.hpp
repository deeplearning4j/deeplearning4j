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

//
// @author raver119@protonmail.com
//
// relies on xoroshiro64** and xoroshiro128 implementations
#include <array/DataTypeUtils.h>
#include <graph/RandomGenerator.h>
#include <helpers/logger.h>
#include <system/op_boilerplate.h>

#include <chrono>

namespace sd {
namespace graph {

template SD_HOST_DEVICE int RandomGenerator::relativeT(sd::LongType, int, int);
template SD_HOST_DEVICE float16 RandomGenerator::relativeT(sd::LongType, float16, float16);
template SD_HOST_DEVICE float RandomGenerator::relativeT(sd::LongType, float, float);
template SD_HOST_DEVICE double RandomGenerator::relativeT(sd::LongType, double, double);
template SD_HOST_DEVICE sd::LongType RandomGenerator::relativeT(sd::LongType, sd::LongType, sd::LongType);

template SD_HOST_DEVICE float16 RandomGenerator::relativeT(sd::LongType);
template SD_HOST_DEVICE float RandomGenerator::relativeT(sd::LongType);
template SD_HOST_DEVICE double RandomGenerator::relativeT(sd::LongType);
}  // namespace graph
}  // namespace sd
