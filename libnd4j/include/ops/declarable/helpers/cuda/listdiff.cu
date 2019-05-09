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
        return saved;
    }

    Nd4jLong listDiffCount(nd4j::LaunchContext * context, NDArray* values, NDArray* keep) {
        auto xType = values->dataType();

        BUILD_SINGLE_SELECTOR(xType, return listDiffCount_, (values, keep), LIBND4J_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template Nd4jLong listDiffCount_, (NDArray* values, NDArray* keep);, LIBND4J_TYPES);

    template <typename T>
    static int listDiffFunctor_(NDArray* values, NDArray* keep, NDArray* output1, NDArray* output2) {
        return Status::OK();
    }

    int listDiffFunctor(nd4j::LaunchContext * context, NDArray* values, NDArray* keep, NDArray* output1, NDArray* output2) {
        auto xType = values->dataType();

        if (DataTypeUtils::isR(xType)) {
            BUILD_SINGLE_SELECTOR(xType, return listDiffFunctor_, (values, keep, output1, output2), FLOAT_TYPES);
        } else if (DataTypeUtils::isZ(xType)) {
            BUILD_SINGLE_SELECTOR(xType, return listDiffFunctor_, (values, keep, output1, output2), INTEGER_TYPES);
        } else {
            throw std::runtime_error("ListDiff: Only integer and floating point data types are supported");
        }
    }

    BUILD_SINGLE_TEMPLATE(template int listDiffFunctor_, (NDArray* values, NDArray* keep, NDArray* output1, NDArray* output2);, FLOAT_TYPES);
    BUILD_SINGLE_TEMPLATE(template int listDiffFunctor_, (NDArray* values, NDArray* keep, NDArray* output1, NDArray* output2);, INTEGER_TYPES);

}
}
}