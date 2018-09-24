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
// This file contains operations added by 3rd parties
//
// @author raver119@gmail.com
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_firas_sparse)

#ifndef LIBND4J_THIRD_PARTY_H
#define LIBND4J_THIRD_PARTY_H

#include <op_boilerplate.h>
#include <memory>
#include <shape.h>
#include <loops/random.h>
#include <NDArray.h>
#include <ops/declarable/DeclarableOp.h>
#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {


        /**
         * This op is special one, and suited only for ProjectionLayer by @firasdib
         *
         *
         *
         * @tparam T
         */
        CUSTOM_OP_IMPL(firas_sparse, 1, 1, false, 0, -1) {
            auto x = INPUT_VARIABLE(0);
            auto z = OUTPUT_VARIABLE(0);

            int batchSize = x->sizeAt(0);
            int numColumns = x->sizeAt(1);

            std::vector<int> indices(*block.getIArguments());
            std::map<int, int> sparse2dense;


            int cnt = 0;
            for (auto v: indices) {
                std::pair<int, int> pair(v, cnt++);
                sparse2dense.insert(pair);
            }

            std::unique_ptr<ResultSet> rows(x->allTensorsAlongDimension({1}));

#pragma omp parallel for schedule(dynamic) proc_bind(close)
            for (int r = 0; r < batchSize; r++) {
                auto row = rows->at(r);

                for (int e = 0; e < numColumns; e += 2) {
                    int idx = row->getIndexedScalar<int>(e);
                    if (idx < 0)
                        break;

                    int denseIdx = sparse2dense.at(idx);


                    float value = row->getIndexedScalar<float>(e + 1);
                    float current = z->e<float>(r, denseIdx);
                    z->putScalar(r, denseIdx, value + current);
                }
            }


            STORE_RESULT(*z);

            return Status::OK();
        }

        DECLARE_SHAPE_FN(firas_sparse) {
            auto inP = inputShape->at(0);

            Nd4jLong *newShape;
            ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(2), Nd4jLong);

            std::vector<Nd4jLong> shape({shape::shapeOf(inP)[0], (Nd4jLong) block.getIArguments()->size()});
            shape::shapeBuffer(2, block.dataType(), shape.data(), newShape);

            return SHAPELIST(newShape);
        }
    }
}

#endif

#endif //LIBND4J_THIRD_PARTY_H
