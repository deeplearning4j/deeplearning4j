//
// @author raver119@gmail.com
//

#ifndef LIBND4J_GNODE_H
#define LIBND4J_GNODE_H

#include <NDArray.h>
#include <graph/generated/node_generated.h>


namespace nd4j {
    namespace graph {

        template<typename T, typename OpName>
        class Node {
        protected:
            DataType _dataType;
            OpType _opType;
            int _opNum;

            // many ops require extra parameters to run
            T *_extraParams;

        public:
            Node(OpType opType);
            Node(const nd4j::graph::FlatNode *node);

            int execute(NDArray<T> *x, NDArray<T> *y, NDArray<T> *z);

            bool equals(Node *other);
        };
    }
}

template<typename T, typename OpName>
nd4j::graph::Node<T,OpName>::Node(OpType opType) {
    this->_opType = opType;
};

template<typename T, typename OpName>
nd4j::graph::Node<T,OpName>::Node(const nd4j::graph::FlatNode *node) {
    if (node != nullptr) {
        this->_dataType = node->dataType();
        this->_opNum = node->opNum();
        this->_opType = node->opType();
    } else {
        // dynamic node, tests probably
    }
}

template<typename T, typename OpName>
bool nd4j::graph::Node<T, OpName>::equals(Node *other) {
    if (_opType == other->_opType && _dataType == other->_dataType && _opNum == other->_opNum)
        return true;

    return false;
}

template<typename T, typename OpName>
int nd4j::graph::Node<T, OpName>::execute(NDArray<T> *x, NDArray<T> *y, NDArray<T> *z) {

    // FIXME: we probably don't want this
    if (z == nullptr)
        z = x;

    switch (this->_opType) {
        case OpType_TRANSFORM: {
                if (y != nullptr && y->nonNull()) {
                    // Pairwise transform
                    //x->template applyPairwiseTransform<OpName>(y, z, this->_extraParams);
                    //functions::pairwise_transforms::PairWiseTransform<T>::template exec<OpName>(x->_buffer, x->_shapeInfo, y->_buffer, y->_shapeInfo, z->_buffer, z->_shapeInfo, this->_extraParams);
                } else {
                    x->template applyTransform<OpName>(z, this->_extraParams);
                }
            }
            break;
        case OpType_ACCUMULATION: {

            }
            break;
        default: {
                throw std::invalid_argument("Unknown OpType is used");
            }
            break;
    }
};
#endif //LIBND4J_GNODE_H
