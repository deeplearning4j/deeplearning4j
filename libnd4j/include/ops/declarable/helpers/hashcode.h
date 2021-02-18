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
// @author raver119@gmail.com
//

#ifndef DEV_TESTS_HASHCODE_H
#define DEV_TESTS_HASHCODE_H

#include "helpers.h"

namespace sd {
    namespace ops {
        namespace helpers {
            template <typename T>
            FORCEINLINE _CUDA_HD Nd4jLong longBytes(T value);

            template <>
            FORCEINLINE _CUDA_HD Nd4jLong longBytes(float value) {
                int intie = *(int *)&value;
                return static_cast<Nd4jLong>(intie);
            }

            template <>
            FORCEINLINE _CUDA_HD Nd4jLong longBytes(double value) {
                Nd4jLong longie = *(Nd4jLong *)&value;
                return longie;
            }

            template <>
            FORCEINLINE _CUDA_HD Nd4jLong longBytes(float16 value) {
                return longBytes<float>((float) value);
            }

            template <>
            FORCEINLINE _CUDA_HD Nd4jLong longBytes(Nd4jLong value) {
                return value;
            }

            template <>
            FORCEINLINE _CUDA_HD Nd4jLong longBytes(bfloat16 value) {
                return longBytes<float>((float) value);
            }

            template <typename T>
            FORCEINLINE _CUDA_HD Nd4jLong longBytes(T value) {
                return longBytes<Nd4jLong>((Nd4jLong) value);
            }


            void hashCode(LaunchContext *context, NDArray &array, NDArray &result);
        }
    }
}

#endif //DEV_TESTS_HASHCODE_H
