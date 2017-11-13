//
// @author raver119@gmail.com
//

#ifndef ND4J_ENUM_UTILS_H
#define ND4J_ENUM_UTILS_H

#include <graph/VariableType.h>
#include <graph/generated/node_generated.h>

namespace nd4j {
    class EnumUtils {
    public:
        static const char * _VariableTypeToString(nd4j::graph::VariableType variableType);
        static const char * _OpTypeToString(nd4j::graph::OpType opType);
        static const char * _LogicOpToString(int opNum);
    };
}

#endif