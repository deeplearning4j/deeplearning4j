/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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
        for (auto array: inputList) {
            length += array->lengthOf();
        }
        auto arrayFull = NDArrayFactory::vector<int>(length, 0, workspace);
        int val = -1;
        Nd4jLong border = 0;
        for (auto& array: inputList) {
            if (array->dataType() != nd4j::DataType::INT32)
                throw std::runtime_error("multiUnique: this op support INT32 data type only.");

            for (Nd4jLong pos = 0; pos < array->lengthOf(); pos++)
                arrayFull->p(border + pos, array->e(pos));
            // memcpy(reinterpret_cast<int*>(arrayFull.buffer() + border), reinterpret_cast<int const*>(array->getBuffer()), array->lengthOf() * array->sizeOf());
            val--;
            border += array->lengthOf();
        }

        nd4j::ops::unique opUnique;
        auto uResult = opUnique.execute({arrayFull}, {}, {}, {});
        if (ND4J_STATUS_OK != uResult->status())
            throw std::runtime_error("multiUnique: cannot execute unique op properly.");

        auto uniqueVals = uResult->at(0);

        bool res = uniqueVals->lengthOf() == arrayFull->lengthOf();

        delete uResult;
        delete arrayFull;
        return res;
    }

}
}
}
