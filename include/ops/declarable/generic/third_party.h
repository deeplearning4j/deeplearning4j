//
// This file contains operations added by 3rd parties
//
// @author raver119@gmail.com
//

#ifndef LIBND4J_THIRD_PARTY_H
#define LIBND4J_THIRD_PARTY_H

#include <op_boilerplate.h>
#include <memory>
#include <shape.h>
#include <ops/ops.h>
#include <loops/random.h>
#include <NDArray.h>
#include <ops/declarable/declarable_ops.h>
#include <NDArrayFactory.h>

namespace nd4j {
    namespace ops {


        /**
         * This op is special one, and suited only for ProjectionLayer by @firasdib
         *
         *
         *
         * @tparam T
         */
        DECLARE_CONFIGURABLE_OP(firas_sparse, 1, 1, false, 0, -1) {
            NDArray<T> *x = block.getVariables().at(0)->getNDArray();
            NDArray<T> *z = this->getZ(block);

            int batchSize = x->sizeAt(0);
            int numColumns = x->sizeAt(1);

            int numIndices = block.getIArguments()->size();

            std::vector<int> indices(*block.getIArguments());
            std::map<int, int> sparse2dense;


            int cnt = 0;
            for (auto v: indices) {
                std::pair<int, int> pair(v, cnt++);
                sparse2dense.insert(pair);
            }

            std::unique_ptr<ArrayList<T>> rows(NDArrayFactory::allTensorsAlongDimension<T>(x, {1}));

#pragma omp parallel for schedule(dynamic) proc_bind(close)
            for (int r = 0; r < batchSize; r++) {
                auto row = rows->at(r);

                for (int e = 0; e < numColumns; e += 2) {
                    int idx = row->getIndexedScalar(e);
                    if (idx < 0)
                        break;

                    int denseIdx = sparse2dense.at(idx);

                    T current = z->getScalar(r, denseIdx);
                    T value = row->getIndexedScalar(e + 1);

                    z->putScalar(r, denseIdx, value);
                }
            }


            STORE_RESULT(*z);

            return ND4J_STATUS_OK;
        }
    }
}

#endif //LIBND4J_THIRD_PARTY_H
