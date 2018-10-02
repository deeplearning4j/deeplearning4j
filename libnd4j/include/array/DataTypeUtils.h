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

#ifndef DATATYPEUTILS_H
#define DATATYPEUTILS_H

#include <types/float16.h>
#include <array/DataType.h>
#include <graph/generated/array_generated.h>
#include <op_boilerplate.h>
#include <dll.h>
#include <Environment.h>
#include <ArrayOptions.h> 
#include <templatemath.h>

namespace nd4j {
    class ND4J_EXPORT DataTypeUtils {
    public:
        static int asInt(DataType type);
        static DataType fromInt(int dtype);
        static DataType fromFlatDataType(nd4j::graph::DataType dtype);

        template <typename T>
        static DataType fromT();
        static size_t sizeOfElement(DataType type);

        // returns the smallest finite value of the given type
        template <typename T>
        FORCEINLINE static _CUDA_HD T min();

        // returns the largest finite value of the given type
        template <typename T>
        FORCEINLINE static _CUDA_HD T max();

        // returns the difference between 1.0 and the next representable value of the given floating-point type 
        template <typename T>
        FORCEINLINE static T eps();

        FORCEINLINE static size_t sizeOf(DataType type);

        FORCEINLINE static bool isR(nd4j::DataType dataType);

        FORCEINLINE static bool isZ(nd4j::DataType dataType);

        FORCEINLINE static bool isB(nd4j::DataType dataType);

        FORCEINLINE static bool isU(nd4j::DataType dataType);

        FORCEINLINE static nd4j::DataType pickPairwiseResultType(nd4j::DataType typeX, nd4j::DataType typeY);

        FORCEINLINE static nd4j::DataType pickPairwiseResultType(const Nd4jLong* shapeInfo1, const Nd4jLong* shapeInfo2);

        FORCEINLINE static nd4j::DataType pickFloatingType(nd4j::DataType typeX);
    };


//////////////////////////////////////////////////////////////////////////
///// IMLEMENTATION OF INLINE METHODS ///// 
//////////////////////////////////////////////////////////////////////////

    FORCEINLINE nd4j::DataType DataTypeUtils::pickFloatingType(nd4j::DataType typeX) {
        // if proposed dataType is already floating point - return it
        if (isR(typeX))
            return typeX;
        else // return default float type otherwise
            return Environment::getInstance()->defaultFloatDataType();
    }

    FORCEINLINE bool DataTypeUtils::isR(nd4j::DataType dataType) {
        return dataType == nd4j::DataType::FLOAT32 || dataType == nd4j::DataType::HALF || dataType == nd4j::DataType::DOUBLE;
    }

    FORCEINLINE bool DataTypeUtils::isB(nd4j::DataType dataType) {
        return dataType == nd4j::DataType::BOOL;
    }

    FORCEINLINE bool DataTypeUtils::isZ(nd4j::DataType dataType) {
        return !isR(dataType) && !isB(dataType);
    }

    FORCEINLINE bool DataTypeUtils::isU(nd4j::DataType dataType) {
        return dataType == nd4j::DataType::UINT8 || dataType == nd4j::DataType::UINT16 || dataType == nd4j::DataType::UINT32 || dataType == nd4j::DataType::UINT64;
    }

    FORCEINLINE nd4j::DataType DataTypeUtils::pickPairwiseResultType(nd4j::DataType typeX, nd4j::DataType typeY) {
        // if both dtypes are the same - just return it
        if (typeX == typeY)
            return typeX;

        auto rX = isR(typeX);
        auto rY = isR(typeY);

        // if X is float - use it
        if (rX && !rY)
            return typeX;

        // if Y is float - use it
        if (!rX && rY)
            return typeY;

        // if both data types are float - return biggest one
        if (rX && rY) {
            // if we allow precision boost, then we pick bigger data type
            if (nd4j::Environment::getInstance()->precisionBoostAllowed()) {
                return nd4j::math::nd4j_max<nd4j::DataType>(typeX, typeY);
            } else {
                // and we return first operand otherwise
                return typeX;
            }

        }

        // if that's not real type, we apply same rules
        if (!rX && !rY) {
            if (nd4j::Environment::getInstance()->precisionBoostAllowed()) {
                return nd4j::math::nd4j_max<nd4j::DataType>(typeX, typeY);
            } else {
                // and we return first operand otherwise
                return typeX;
            }
        }

        return typeX;
    }

///////////////////////////////////////////////////////////////////
FORCEINLINE nd4j::DataType DataTypeUtils::pickPairwiseResultType(const Nd4jLong* shapeInfo1, const Nd4jLong* shapeInfo2) {

    return pickPairwiseResultType(ArrayOptions::dataType(shapeInfo1), ArrayOptions::dataType(shapeInfo2));
}

///////////////////////////////////////////////////////////////////
FORCEINLINE size_t DataTypeUtils::sizeOf(DataType type) {
    return sizeOfElement(type);
}

// returns the smallest finite value of the given type
template<>
FORCEINLINE _CUDA_HD int DataTypeUtils::min<int>() {
    return 1;
}

template<>
FORCEINLINE _CUDA_HD char DataTypeUtils::min<char>() {
    return 1;
}

template <>
FORCEINLINE _CUDA_HD bool DataTypeUtils::min<bool>() {
    return false;
}

template<>
FORCEINLINE _CUDA_HD Nd4jLong DataTypeUtils::min<Nd4jLong>() {
    return 1L;
}

template<>
FORCEINLINE _CUDA_HD uint64_t DataTypeUtils::min<uint64_t>() {
    return 1L;
}

template<>
FORCEINLINE _CUDA_HD uint32_t DataTypeUtils::min<uint32_t>() {
    return 1;
}

template<>
FORCEINLINE _CUDA_HD float DataTypeUtils::min<float>() {         
    return 1.175494e-38;    
}

template<>
FORCEINLINE _CUDA_HD float16 DataTypeUtils::min<float16>() {
    return (float16) 6.1035e-05;    
}

template<>
FORCEINLINE _CUDA_HD double DataTypeUtils::min<double>() {       
    return 2.2250738585072014e-308;    
}

///////////////////////////////////////////////////////////////////
// returns the largest finite value of the given type
template <>
FORCEINLINE _CUDA_HD int DataTypeUtils::max<int>() {
    return 2147483647;
}

template <>
FORCEINLINE _CUDA_HD bool DataTypeUtils::max<bool>() {
    return true;
}

template <>
FORCEINLINE _CUDA_HD int8_t DataTypeUtils::max<int8_t>() {
    return 127;
}

template <>
FORCEINLINE _CUDA_HD uint8_t DataTypeUtils::max<uint8_t>() {
    return 255;
}

template <>
FORCEINLINE _CUDA_HD int16_t DataTypeUtils::max<int16_t>() {
    return 32767;
}

template <>
FORCEINLINE _CUDA_HD uint16_t DataTypeUtils::max<uint16_t>() {
    return 65535;
}

template <>
FORCEINLINE _CUDA_HD Nd4jLong DataTypeUtils::max<Nd4jLong>() {
    return 9223372036854775807LL;
}

template <>
FORCEINLINE _CUDA_HD uint32_t DataTypeUtils::max<uint32_t>() {
    return 4294967295;
}

template <>
FORCEINLINE _CUDA_HD Nd4jULong DataTypeUtils::max<Nd4jULong>() {
    return 18446744073709551615LLU;
}

template <>
FORCEINLINE _CUDA_HD float DataTypeUtils::max<float>() {    
    return 3.402823e+38;
}

template <>
FORCEINLINE _CUDA_HD double DataTypeUtils::max<double>() {       
    return 1.7976931348623157E308;   
}

template <>
FORCEINLINE _CUDA_HD float16 DataTypeUtils::max<float16>() {       
    return static_cast<float16>(65504.f);
}

///////////////////////////////////////////////////////////////////
// returns the difference between 1.0 and the next representable value of the given floating-point type 
template <typename T>
FORCEINLINE T DataTypeUtils::eps() {
        if (std::is_same<T, double>::value)
            return std::numeric_limits<double>::epsilon();
        else if (std::is_same<T, float>::value)
            return std::numeric_limits<float>::epsilon();
        else if (std::is_same<T, float16>::value)
            return 0.00097656;    
        else
            return 0;
}


}

#endif //DATATYPEUTILS_H