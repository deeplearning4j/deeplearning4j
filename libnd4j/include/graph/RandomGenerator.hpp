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

#include <system/op_boilerplate.h>
#include <system/pointercast.h>
#include <graph/RandomGenerator.h>
#include <chrono>
#include <array/DataTypeUtils.h>
#include <helpers/logger.h>

namespace sd {
    namespace graph {



        template _CUDA_HD int RandomGenerator::relativeT(Nd4jLong, int, int);
        template _CUDA_HD float16 RandomGenerator::relativeT(Nd4jLong, float16, float16);
        template _CUDA_HD float RandomGenerator::relativeT(Nd4jLong, float, float);
        template _CUDA_HD double RandomGenerator::relativeT(Nd4jLong, double, double);
        template _CUDA_HD Nd4jLong RandomGenerator::relativeT(Nd4jLong, Nd4jLong, Nd4jLong);

        template _CUDA_HD float16 RandomGenerator::relativeT(Nd4jLong);
        template _CUDA_HD float RandomGenerator::relativeT(Nd4jLong);
        template _CUDA_HD double RandomGenerator::relativeT(Nd4jLong);
    }
}