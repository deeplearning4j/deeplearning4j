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
// Created by raver119 on 22.11.2017.
//

#include <graph/FlatUtils.h>
#include <array/DataTypeConversions.h>
#include <array/DataTypeUtils.h>
#include <array/ByteOrderUtils.h>


namespace nd4j {
    namespace graph {
        std::pair<int, int> FlatUtils::fromIntPair(IntPair *pair) {
            return std::pair<int, int>(pair->first(), pair->second());
        }

        std::pair<Nd4jLong, Nd4jLong> FlatUtils::fromLongPair(LongPair *pair) {
            return std::pair<Nd4jLong, Nd4jLong>(pair->first(), pair->second());
        }

        template<typename T>
        NDArray<T> *FlatUtils::fromFlatArray(const nd4j::graph::FlatArray *flatArray) {
            auto rank = static_cast<int>(flatArray->shape()->Get(0));
            auto newShape = new Nd4jLong[shape::shapeInfoLength(rank)];
            memcpy(newShape, flatArray->shape()->data(), shape::shapeInfoByteLength(rank));

            // empty arrays is special case, nothing to restore here
            if (shape::isEmpty(newShape)) {
                delete[] newShape;
                return NDArray<T>::createEmpty();
            }

            auto length = shape::length(newShape);
            auto newBuffer = new T[length];
            auto dtype = DataTypeUtils::fromFlatDataType(flatArray->dtype());

            DataTypeConversions<T>::convertType(newBuffer, (void *)flatArray->buffer()->data(), dtype, ByteOrderUtils::fromFlatByteOrder(flatArray->byteOrder()),  length);

            auto array = new NDArray<T>(newBuffer, newShape);
            array->triggerAllocationFlag(true, true);

            return array;
        }


        template NDArray<float> *FlatUtils::fromFlatArray<float>(const nd4j::graph::FlatArray *flatArray);
        template NDArray<float16> *FlatUtils::fromFlatArray<float16>(const nd4j::graph::FlatArray *flatArray);
        template NDArray<double> *FlatUtils::fromFlatArray<double>(const nd4j::graph::FlatArray *flatArray);
        template NDArray<int> *FlatUtils::fromFlatArray<int>(const nd4j::graph::FlatArray *flatArray);
        template NDArray<Nd4jLong> *FlatUtils::fromFlatArray<Nd4jLong>(const nd4j::graph::FlatArray *flatArray);
    }
}