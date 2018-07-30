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

namespace nd4j {
    class DataTypeUtils {
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
        
    };


//////////////////////////////////////////////////////////////////////////
///// IMLEMENTATION OF INLINE METHODS ///// 
//////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////
// returns the smallest finite value of the given type
template<>
FORCEINLINE _CUDA_HD int DataTypeUtils::min<int>() {
    return 1;
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
FORCEINLINE _CUDA_HD Nd4jLong DataTypeUtils::max<Nd4jLong>() {
    return 9223372036854775807LL;
}

template <>
FORCEINLINE _CUDA_HD uint64_t DataTypeUtils::max<uint64_t>() {
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
    return (float16) 65504.f;   
}

///////////////////////////////////////////////////////////////////
// returns the difference between 1.0 and the next representable value of the given floating-point type 
template <typename T>
FORCEINLINE T DataTypeUtils::eps() {

	switch (sizeof(T)) {
        case 8:                // T = double            
            return std::numeric_limits<double>::epsilon();    
        case 4:                // T = float            
            return std::numeric_limits<float>::epsilon();    
        case 2:                // T = float16        
            return 0.00097656;    
        default:
            throw("DataTypeUtils::epsilon function: type of T is undefined !");    
        }
}


}

#endif //DATATYPEUTILS_H