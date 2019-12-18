/*******************************************************************************
 * Copyright (c) 2019 Konduit K.K.
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
 
#include <ops/declarable/helpers/adjust_hue.h>
#include <ops/declarable/helpers/color_models_conv.h>
#include <helpers/ConstantTadHelper.h>
#include <execution/Threads.h>

namespace nd4j {
    namespace ops {
        namespace helpers {

            //local
            template <typename T, typename Op>
            FORCEINLINE static void triple_transformer(const NDArray* input, NDArray* output, const int dimC, Op op) {

                const int rank = input->rankOf();

                const T* x = input->bufferAsT<T>();
                T* z = output->bufferAsT<T>();

                if (dimC == rank - 1 && input->ews() == 1 && output->ews() == 1 && input->ordering() == 'c' && output->ordering() == 'c') {

                    auto func = PRAGMA_THREADS_FOR{
                        for (auto i = start; i < stop; i += increment) {
                            op(x[i], x[i + 1], x[i + 2], z[i], z[i + 1], z[i + 2]);
                        }
                    };

                    samediff::Threads::parallel_for(func, 0, input->lengthOf(), 3);
                }
                else {
                    auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(input->getShapeInfo(), dimC);
                    auto packZ = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(output->getShapeInfo(), dimC);

                    const Nd4jLong numOfTads = packX.numberOfTads();
                    const Nd4jLong xDimCstride = input->stridesOf()[dimC];
                    const Nd4jLong zDimCstride = output->stridesOf()[dimC];

                    auto func = PRAGMA_THREADS_FOR{
                        for (auto i = start; i < stop; i += increment) {
                            const T* xTad = x + packX.platformOffsets()[i];
                            T* zTad = z + packZ.platformOffsets()[i];
                            op(xTad[0], xTad[xDimCstride], xTad[2 * xDimCstride], zTad[0], zTad[zDimCstride], zTad[2 * zDimCstride]);

                        }
                    };

                    samediff::Threads::parallel_tad(func, 0, numOfTads);
                }
            }



            template <typename T>
            FORCEINLINE static void hsv_rgb(const NDArray* input, NDArray* output, const int dimC) {
                auto op = nd4j::ops::helpers::hsvToRgb<T>;
                return triple_transformer<T>(input, output, dimC, op);
            }

            template <typename T>
            FORCEINLINE static void rgb_hsv(const NDArray* input, NDArray* output, const int dimC) {
                auto op = nd4j::ops::helpers::rgbToHsv<T>;
                return triple_transformer<T>(input, output, dimC, op);
            }

            void transform_hsv_rgb(nd4j::LaunchContext* context, const NDArray* input, NDArray* output, const int dimC) {
                BUILD_SINGLE_SELECTOR(input->dataType(), hsv_rgb, (input, output, dimC), FLOAT_TYPES);
            }

            void transform_rgb_hsv(nd4j::LaunchContext* context, const NDArray* input, NDArray* output, const int dimC) {
                BUILD_SINGLE_SELECTOR(input->dataType(), rgb_hsv, (input, output, dimC), FLOAT_TYPES);
            }

        }
    }
}