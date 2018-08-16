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
// @author raver119@gmail.com
//

#include <helpers/IRandomGenerator.h>
#include <array/DataTypeUtils.h>

namespace nd4j {

    template <>
    int IRandomGenerator::relativeT<int>(Nd4jLong index) {
        return this->relativeInt(index);
    }

    template <>
    uint32_t IRandomGenerator::relativeT<uint32_t>(Nd4jLong index) {
        return this->relativeUint32(index);
    }

    template <>
    uint64_t IRandomGenerator::relativeT<uint64_t>(Nd4jLong index) {
        return this->relativeUint64(index);
    }

    template <>
    Nd4jLong IRandomGenerator::relativeT<Nd4jLong>(Nd4jLong index) {
        return this->relativeLong(index);
    }

    template <typename T>
    T IRandomGenerator::relativeT(Nd4jLong index) {
        // This is default implementation for floating point types
        auto i = static_cast<float>(this->relativeInt(index));
        auto r = i / static_cast<float>(DataTypeUtils::max<int>());
        return static_cast<T>(r);
    }

    template <typename T>
    T IRandomGenerator::relativeT(Nd4jLong index, T from, T to) {
        return from + (this->relativeT<T>(index) * (to - from));
    }


    int IRandomGenerator::relativeInt(Nd4jLong index) {
        auto x = this->relativeUint32(index);
        auto r = x < DataTypeUtils::max<int>() ? x : static_cast<int>(x % DataTypeUtils::max<int>());
        return r;
    }

    Nd4jLong IRandomGenerator::relativeLong(Nd4jLong index) {
        auto x = this->relativeUint64(index);
        auto r = x < DataTypeUtils::max<Nd4jLong>() ? x :static_cast<int>(x % DataTypeUtils::max<Nd4jLong>());
        return r;
    }


    template float IRandomGenerator::relativeT<float>(Nd4jLong index);
    template float16 IRandomGenerator::relativeT<float16>(Nd4jLong index);
    template double IRandomGenerator::relativeT<double>(Nd4jLong index);

    template float IRandomGenerator::relativeT<float>(Nd4jLong index, float from, float to);
    template float16 IRandomGenerator::relativeT<float16>(Nd4jLong index, float16 from, float16 to);
    template double IRandomGenerator::relativeT<double>(Nd4jLong index, double from, double to);
    template int IRandomGenerator::relativeT<int>(Nd4jLong index, int from, int to);
    template Nd4jLong IRandomGenerator::relativeT<Nd4jLong>(Nd4jLong index, Nd4jLong from, Nd4jLong to);

}
