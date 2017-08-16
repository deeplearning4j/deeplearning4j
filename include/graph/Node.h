//
// @author raver119@gmail.com
//

#ifndef LIBND4J_GNODE_H
#define LIBND4J_GNODE_H

#include <atomic>
#include <NDArray.h>
#include <graph/generated/node_generated.h>


namespace nd4j {
    namespace graph {
        class Node {
        protected:
            DataType _dataType;
            OpType _opType;
            int _opNum;
            int _id;
            std::vector<int> _input;
            std::vector<int> _output;

            std::atomic<int> _finished;

            // this variable points to onion layer within graph
            int _layer = -1;

            // many ops require extra parameters to run
            float *_extraParams;

            bool _hasExternalOutputs;
            bool _hasExternalInputs;
            bool _hasInternalOutputs;
            bool _hasInternalInputs;

        public:
            Node(OpType opType = OpType_TRANSFORM, int opNum = 0, int id = 0, std::initializer_list<int> input = {}, std::initializer_list<int> output = {});
            Node(const nd4j::graph::FlatNode *node);
            ~Node();

            bool equals(Node *other);

            OpType opType();
            int opNum();
            int id();
            std::vector<int> *input();
            std::vector<int> *output();

            float *extraParams();

            bool isMultiInput();
            bool isMultiOutput();

            int getLayer();
            void setLayer(int layer);

            bool hasExternalOutputs();
            bool hasExternalInputs();
            bool hasInternalOutputs();
            bool hasInternalInputs();

            void prepare();
            void finished();
            void waitTillFinished();
        };
    }
}

int nd4j::graph::Node::getLayer() {
    return _layer;
}
void nd4j::graph::Node::setLayer(int layer) {
    _layer = layer;
}

bool nd4j::graph::Node::hasExternalOutputs() {
    return _hasExternalOutputs;
}

bool nd4j::graph::Node::hasExternalInputs() {
    return _hasExternalInputs;
}

bool nd4j::graph::Node::hasInternalOutputs() {
    return _hasInternalOutputs;
}

bool nd4j::graph::Node::hasInternalInputs() {
    return _hasInternalInputs;
}

void nd4j::graph::Node::finished() {
    _finished.store(1);
}

void nd4j::graph::Node::prepare() {
    _finished.store(0);
}

void nd4j::graph::Node::waitTillFinished() {
    while (_finished != 1)
        usleep(10);
}

bool nd4j::graph::Node::isMultiInput() {
    return _input.size() > 1;
}

bool nd4j::graph::Node::isMultiOutput() {
    return _output.size() > 1;
}

float * nd4j::graph::Node::extraParams() {
    return _extraParams;
}

nd4j::graph::OpType nd4j::graph::Node::opType() {
    return _opType;
}

int nd4j::graph::Node::id() {
    return _id;
}

int nd4j::graph::Node::opNum() {
    return _opNum;
}

std::vector<int> *nd4j::graph::Node::input() {
    return &_input;
}

std::vector<int> *nd4j::graph::Node::output() {
    return &_output;
}

nd4j::graph::Node::Node(OpType opType, int opNum, int id, std::initializer_list<int> input, std::initializer_list<int> output) {
    this->_finished.store(0);
    this->_opType = opType;
    this->_id = id;
    this->_opNum = opNum;

    _hasExternalInputs = false;
    _hasExternalOutputs = false;
    _hasInternalInputs = false;
    _hasInternalOutputs = false;

    for (auto i: input) {
        _input.push_back(i);
        if (i < 0)
            _hasExternalInputs = true;
        else
            _hasInternalInputs = true;

    }


    for (auto o: output) {
        _output.push_back(o);
        if (o < 0)
            _hasExternalOutputs = true;
        else
            _hasInternalOutputs = true;
    }

};

nd4j::graph::Node::Node(const nd4j::graph::FlatNode *node) {
    _finished.store(0);

    _hasExternalInputs = false;
    _hasExternalOutputs = false;
    _hasInternalInputs = false;
    _hasInternalOutputs = false;

    if (node != nullptr) {
        this->_id = node->id();
        this->_dataType = node->dataType();
        this->_opNum = node->opNum();
        this->_opType = node->opType();

        if (node->input() != nullptr)
            for (int e = 0; e < node->input()->size(); e++) {
                _input.push_back(node->input()->Get(e));

                if (node->input()->Get(e) < 0)
                    _hasExternalInputs = true;
                else
                    _hasInternalInputs = true;
            }

        if (node->output() != nullptr)
            for (int e = 0; e < node->output()->size(); e++) {
                _output.push_back(node->output()->Get(e));

                if (node->output()->Get(e) < 0)
                    _hasExternalOutputs = true;
                else
                    _hasInternalOutputs = true;
            }

        if (node->extraParams() != nullptr && node->extraParams()->size()) {
            _extraParams = new float[node->extraParams()->size()];
            for (int e = 0; e < node->extraParams()->size(); e++) {
                _extraParams[e] = node->extraParams()->Get(e);
            }
        }


    } else {
        // empty dynamic node, tests probably
    }
}

nd4j::graph::Node::~Node() {
    if (_extraParams != nullptr)
        delete[] _extraParams;
}

bool nd4j::graph::Node::equals(Node *other) {
    if (_opType == other->_opType && _dataType == other->_dataType && _opNum == other->_opNum)
        return true;

    return false;
}


#endif //LIBND4J_GNODE_H
