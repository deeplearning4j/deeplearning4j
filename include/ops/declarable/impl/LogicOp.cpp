//
// Created by raver119 on 15.10.2017.
//

#include "ops/declarable/LogicOp.h"

namespace nd4j {
    namespace ops {

        template <typename T>
        LogicOp<T>::LogicOp(const char *name) : DeclarableOp<T>::DeclarableOp(name, true) {
            // just using DeclarableOp constructor
        }

        template <typename T>
        Nd4jStatus LogicOp<T>::validateAndExecute(nd4j::graph::Block<T> &block) {
            nd4j_logger("WARNING: LogicOps should NOT be ever called\n", "");
            return ND4J_STATUS_BAD_INPUT;
        }

        template <typename T>
        ShapeList* LogicOp<T>::calculateOutputShape(ShapeList *inputShape, nd4j::graph::Block<T> &block) {
            // FIXME: we probably want these ops to evaluate scopes
            return new ShapeList();
        }

        template class LogicOp<float>;
        template class LogicOp<float16>;
        template class LogicOp<double>;
    }
}