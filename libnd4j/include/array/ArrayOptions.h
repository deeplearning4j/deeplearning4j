//
// @author raver119@gmail.com
//

#ifndef ND4J_ARRAY_OPTIONS_H
#define ND4J_ARRAY_OPTIONS_H

#include <op_boilerplate.h>
#include <pointercast.h>
#include <dll.h>
#include <array/DataType.h>
#include <array/ArrayType.h>
#include <array/SpaceType.h>
#include <array/SparseType.h>
#include <initializer_list>


#define ARRAY_SPARSE 2
#define ARRAY_COMPRESSED 4

#define ARRAY_CSR 16
#define ARRAY_CSC 32
#define ARRAY_COO 64


// complex values
#define ARRAY_COMPLEX 512

// quantized values
#define ARRAY_QUANTIZED 1024


//  16 bit float
#define ARRAY_HALF 4096

// 16 bit float that uses 32 bits of memort (heym, CUDA!)
#define ARRAY_HALF2 8192

// regular 32 bit float
#define ARRAY_FLOAT 16384

// regular 64 biy float
#define ARRAY_DOUBLE 32768

// 8 bit integer
#define ARRAY_CHAR 65536

// 16 bit integer
#define ARRAY_SHORT 131072

// 32 bit integer
#define ARRAY_INT 262144

// 64 bit integer
#define ARRAY_LONG 524288

// flag for extras 
#define ARRAY_EXTRAS 2097152


// flag for signed/unsigned integers
#define ARRAY_UNSIGNED 8388608


namespace nd4j {
    class ND4J_EXPORT ArrayOptions {

    public:
        static bool isNewFormat(Nd4jLong *shapeInfo);
        static bool hasPropertyBitSet(Nd4jLong *shapeInfo, int property);
        static bool togglePropertyBit(Nd4jLong *shapeInfo, int property);
        static void unsetPropertyBit(Nd4jLong *shapeInfo, int property);
        static void setPropertyBit(Nd4jLong *shapeInfo, int property);
        static void setPropertyBits(Nd4jLong *shapeInfo, std::initializer_list<int> properties);

        static bool isSparseArray(Nd4jLong *shapeInfo);
        static bool isUnsigned(Nd4jLong *shapeInfo);

        static nd4j::DataType dataType(Nd4jLong *shapeInfo);
        static SpaceType spaceType(Nd4jLong *shapeInfo);
        static ArrayType arrayType(Nd4jLong *shapeInfo);
        static SparseType sparseType(Nd4jLong *shapeInfo);

        static bool hasExtraProperties(Nd4jLong *shapeInfo);
    };
}

#endif // ND4J_ARRAY_OPTIONS_H :)