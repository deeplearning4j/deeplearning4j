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

#include <ops/declarable/helpers/multiUnique.h>
#include <ops/declarable/CustomOperations.h>

namespace nd4j {
namespace ops {
namespace helpers {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    bool multiUnique(std::vector<NDArray*> const& inputList, nd4j::memory::Workspace *workspace) {
        Nd4jLong length = 0;
        std::vector<NDArray> reshaped(inputList.size());
        int pos = 0;
        Nd4jLong axis = 0;
        Context cContext(1);
        for (auto array: inputList) {
            if (array->dataType() != nd4j::DataType::INT32)
                throw std::runtime_error("multiUnique: this op support INT32 data type only.");

            reshaped[pos] = array->reshape(array->ordering(), {-1});
            cContext.setInputArray(pos, &reshaped[pos]);

            length += array->lengthOf();
            pos++;
        }
        NDArray arrayFull('c', {length}, nd4j::DataType::INT32);
        cContext.setOutputArray(0, &arrayFull);
        cContext.setIArguments(&axis, 1);

        nd4j::ops::concat opConcat;
        auto cResult = opConcat.execute(&cContext);
        if (Status::OK() != cResult)
            throw std::runtime_error("multiUnique: cannot execute concat op properly.");

        nd4j::ops::unique opUnique;
        auto uResult = opUnique.evaluate({&arrayFull});
        if (Status::OK() != uResult->status())
            throw std::runtime_error("multiUnique: cannot execute unique op properly.");

        auto uniqueVals = uResult->at(0);

        bool res = uniqueVals->lengthOf() == arrayFull.lengthOf();

        delete uResult;        
        return res;
    }

}
}
}
