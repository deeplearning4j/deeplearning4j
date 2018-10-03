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
//  @author sgazeos@gmail.com
//

#include <ops/declarable/helpers/listdiff.h>
#include <vector>
//#include <memory>

namespace nd4j {
namespace ops {
namespace helpers {
    template <typename T>
    static Nd4jLong listDiffCount_(NDArray* values, NDArray* keep) {

        Nd4jLong saved = 0L;
        for (Nd4jLong e = 0; e < values->lengthOf(); e++) {
            T v = values->e<T>(e);
            T extras[] = {v, (T) 0.0f, (T) 10.0f};
            NDArray idx = keep->indexReduceNumber(indexreduce::FirstIndex, extras);
            Nd4jLong index = idx.e<Nd4jLong>(0);
            if (index < 0)
                saved++;
        }
        return saved;
    }

    Nd4jLong listDiffCount(NDArray* values, NDArray* keep) {
        auto xType = values->dataType();

        BUILD_SINGLE_SELECTOR(xType, return listDiffCount_, (values, keep), FLOAT_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template Nd4jLong listDiffCount_, (NDArray* values, NDArray* keep);, FLOAT_TYPES);

    template <typename T>
    static int listDiffFunctor_(NDArray* values, NDArray* keep, NDArray* output1, NDArray* output2) {

        std::vector<T> saved;
        std::vector<Nd4jLong> indices;


        for (Nd4jLong e = 0; e < values->lengthOf(); e++) {
            T v = values->e<T>(e);
            T extras[] = {v, (T) 0.0f, (T) 10.0f};
            NDArray idxScalar = keep->indexReduceNumber(indexreduce::FirstIndex, extras);
            Nd4jLong idx = idxScalar.e<Nd4jLong>(0);

            if (idx < 0) {
                saved.emplace_back(v);
                indices.emplace_back(e);
            }
        }


        if (saved.size() == 0) {
//            if (nd4j::ops::conditionHelper(__FILE__, __LINE__, false, 0, "ListDiff: search returned no results") != 0)
            nd4j_printf("ListDiff: search returned no results", "");
                throw std::invalid_argument("Op validation failed");
        } else {
            auto z0 = output1;//OUTPUT_VARIABLE(0); //new NDArray<T>('c', {(int) saved.size()});
            auto z1 = output2; //OUTPUT_VARIABLE(1); //new NDArray<T>('c', {(int) saved.size()});

            if (z0->lengthOf() != saved.size()) {
                nd4j_printf("ListDiff: output/actual size mismatch", "");
                throw std::invalid_argument("Op validation failed");
            }

            if (z1->lengthOf() != saved.size()) {
                nd4j_printf("ListDiff: output/actual indices size mismatch", "");
                throw std::invalid_argument("Op validation failed");
            }
//            if (nd4j::ops::conditionHelper(__FILE__, __LINE__, z0->lengthOf() == saved.size(), 0, "ListDiff: output/actual size mismatch") != 0)
//                throw std::invalid_argument("Op validation failed");
//            if (nd4j::ops::conditionHelper(__FILE__, __LINE__, z1->lengthOf() == saved.size(), 0, "ListDiff: output/actual indices size mismatch") != 0)
//                throw std::invalid_argument("Op validation failed");
//            REQUIRE_TRUE(z0->lengthOf() == saved.size(), 0, "ListDiff: output/actual size mismatch");
//            REQUIRE_TRUE(z1->lengthOf() == saved.size(), 0, "ListDiff: output/actual size mismatch");

            memcpy(z0->buffer(), saved.data(), saved.size() * sizeof(T));
            memcpy(z1->buffer(), indices.data(), indices.size() * sizeof(Nd4jLong));

            //OVERWRITE_2_RESULTS(z0, z1);
    //        STORE_2_RESULTS(z0, z1);
        }
        return ND4J_STATUS_OK;
    }

    int listDiffFunctor(NDArray* values, NDArray* keep, NDArray* output1, NDArray* output2) {
        auto xType = values->dataType();

        BUILD_SINGLE_SELECTOR(xType, return listDiffFunctor_, (values, keep, output1, output2), FLOAT_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template int listDiffFunctor_, (NDArray* values, NDArray* keep, NDArray* output1, NDArray* output2);, FLOAT_TYPES);

}
}
}