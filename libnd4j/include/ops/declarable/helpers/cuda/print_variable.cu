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

//
//  @author raver119@gmail.com
//

#include <ops/declarable/helpers/print_variable.h>
#include <helpers/PointersManager.h>

namespace sd {
    namespace ops {
        namespace helpers {
            template <typename T>
            static _CUDA_G void print_device(const void *special, const Nd4jLong *shapeInfo) {
                auto length = shape::length(shapeInfo);
                auto x = reinterpret_cast<const T*>(special);

                // TODO: add formatting here
                printf("[");

                for (uint64_t e = 0; e < length; e++) {
                    printf("%f", (float) x[shape::getIndexOffset(e, shapeInfo)]);

                    if (e < length - 1)
                        printf(", ");
                }

                printf("]\n");
            }

            template <typename T>
            static _CUDA_H void exec_print_device(LaunchContext &ctx, const void *special, const Nd4jLong *shapeInfo) {
                print_device<T><<<1, 1, 1024, *ctx.getCudaStream()>>>(special, shapeInfo);
            }

            void print_special(LaunchContext &ctx, const NDArray &array, const std::string &message) {
                NDArray::prepareSpecialUse({}, {&array});

                PointersManager pm(&ctx, "print_device");
                BUILD_SINGLE_SELECTOR(array.dataType(), exec_print_device, (ctx, array.specialBuffer(), array.specialShapeInfo()), LIBND4J_TYPES)
                pm.synchronize();

                NDArray::registerSpecialUse({}, {&array});
            }
        }
    }
}
