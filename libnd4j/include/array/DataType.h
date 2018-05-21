//
// @author raver119@gmail.com
//

#ifndef ND4J_DATATYPE_H
#define ND4J_DATATYPE_H

namespace nd4j {
    enum DataType {
        DataType_INHERIT = 0,
        DataType_BOOL = 1,
        DataType_FLOAT8 = 2,
        DataType_HALF = 3,
        DataType_HALF2 = 4,
        DataType_FLOAT = 5,
        DataType_DOUBLE = 6,
        DataType_INT8 = 7,
        DataType_INT16 = 8,
        DataType_INT32 = 9,
        DataType_INT64 = 10,
        DataType_UINT8 = 11,
        DataType_UINT16 = 12,
        DataType_UINT32 = 13,
        DataType_UINT64 = 14,
        DataType_QINT8 = 15,
        DataType_QINT16 = 16,
    };
}

#endif