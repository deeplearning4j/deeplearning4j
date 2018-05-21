//
// @author raver119@gmail.com
//

#include <graph/VariableType.h>
#include <helpers/EnumUtils.h>

using namespace nd4j::graph;

namespace nd4j {
    const char * EnumUtils::_VariableTypeToString(nd4j::graph::VariableType variableType) {
        switch (variableType) {
            case NDARRAY: return "NDARRAY";
            case ARRAY_LIST: return "ARRAY_LIST";
            case FLOW: return "FLOW";
            default: return "UNKNOWN VariableType";
        }
    }

    const char * EnumUtils::_OpTypeToString(nd4j::graph::OpType opType) {
        switch(opType) {
            case OpType_ACCUMULATION: return "ACCUMULATION";
            case OpType_BOOLEAN: return "BOOLEAN";
            case OpType_BROADCAST: return "BROADCAST";
            case OpType_CUSTOM: return "CUSTOM";
            case OpType_LOGIC: return "LOGIC";
            case OpType_TRANSFORM: return "TRANSFORM";
            case OpType_INDEX_ACCUMULATION: return "INDEX_ACCUMULATION";
            case OpType_SCALAR: return "SCALAR";
            case OpType_SHAPE: return "SHAPE";
            default: return "UNKNOWN OpType";
        }
    }


    const char * EnumUtils::_LogicOpToString(int opNum) {
        switch(opNum) {
            case 0: return "WHILE";
            case 10: return "SCOPE";
            case 20: return "CONDITIONAL";
            case 30: return "SWITCH";
            case 40: return "RETURN";
            case 60: return "MERGE";
            case 70: return "LOOP_COND";
            case 80: return "NEXT_ITERATION";
            case 90: return "EXIT";
            case 100: return "ENTER";
            default: return "UNKNOWN OPERATION";
        } 
    }
}