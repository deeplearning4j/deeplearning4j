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
// Created by raver119 on 24/09/18.
//

#include <ops/declarable/helpers/where.h>
#include <array/NDArrayList.h>

namespace sd {
    namespace ops {
        namespace helpers {
            template <typename T>
            static void __where(NDArray &condition, NDArray& output, memory::Workspace *workspace) {
                NDArrayList list(0, true);
                int cnt = 0;

                int idx[MAX_RANK];

                for (Nd4jLong e = 0; e < condition.lengthOf(); e++) {

                    shape::index2coordsCPU(0, e, condition.getShapeInfo(), idx);

                    auto offset = shape::getOffset(condition.getShapeInfo(), idx);

                    if (condition.e<bool>(offset)) {
                        auto array = NDArrayFactory::create_('c', {1, condition.rankOf()}, output.dataType(), output.getContext());
                        for (int f = 0; f < condition.rankOf(); f++)
                            array->p(f, (T) idx[f]);

                        list.write(cnt++, array);
                    }
                }

                auto s = list.stack();
                output.assign(s);
                delete s;
            }
            BUILD_SINGLE_TEMPLATE(template void __where,(NDArray &condition, NDArray& output, memory::Workspace *workspace), LIBND4J_TYPES);

            void _where(sd::LaunchContext * context, NDArray &condition, NDArray& output, memory::Workspace *workspace) {
                condition.syncToHost();
                BUILD_SINGLE_SELECTOR(output.dataType(), __where, (condition, output, workspace), LIBND4J_TYPES);
                output.syncToDevice();
            }
        }
    }
}
