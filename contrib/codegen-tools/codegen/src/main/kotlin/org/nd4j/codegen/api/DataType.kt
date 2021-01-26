package org.nd4j.codegen.api

enum class DataType {
    NDARRAY,        // Any NDArray type (input only) - INDArray or SDVariable
    FLOATING_POINT, // Any floating point data type
    INT, // integer data type
    LONG, //long, signed int64 datatype
    NUMERIC, // any floating point or integer data type
    BOOL, // boolean data type
    STRING, //String value
    // Arg only
    DATA_TYPE, // tensor data type
    CONDITION, // A condition
    LOSS_REDUCE, // Loss reduction mode
    ENUM; // defines an enum along with possibleValues property in Arg

    fun isTensorDataType() = setOf(NDARRAY, FLOATING_POINT, INT, LONG, NUMERIC, BOOL).contains(this)
}