//
// @author raver119@gmail.com
//

#ifndef LIBND4J_GNODE_H
#define LIBND4J_GNODE_H

#include <graph/generated/node_generated.h>


namespace nd4j {
    namespace graph {

        template<typename T, typename Op>
        class GNode {
        protected:
            DataType _dataType;
            OpType _opType;
            int _opNum;

        public:
            GNode(const nd4j::graph::FlatNode *node);


            bool equals(GNode *other);
        };
    }
}

template<typename T, typename Op>
nd4j::graph::GNode<T,Op>::GNode(const nd4j::graph::FlatNode *node) {
    this->_dataType = node->dataType();
    this->_opNum = node->opNum();
    this->_opType = node->opType();
}

template<typename T, typename Op>
bool nd4j::graph::GNode<T, Op>::equals(GNode *other) {
    if (_opType == other->_opType && _dataType == other->_dataType && _opNum == other->_opNum)
        return true;

    return false;
}

#endif //LIBND4J_GNODE_H
