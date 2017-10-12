//
// @author raver119@gmail.com
//

#include <graph/Node.h>
#include <ops/declarable/OpRegistrator.h>

namespace nd4j {
    namespace graph {

        template <typename T>
        void nd4j::graph::Node<T>::setOuterTime(Nd4jIndex time){
            if (hasBlockAttached())
                _block->setOuterTime(time);
        }

        template <typename T>
        void nd4j::graph::Node<T>::setInnerTime(Nd4jIndex time){
            if (hasBlockAttached())
                _block->setInnerTime(time);
        }

        template <typename T>
        void nd4j::graph::Node<T>::setGraph(nd4j::graph::Graph<T>* graph) {
            _graph = graph;
        }

        template <typename T>
        nd4j::graph::Graph<T>* nd4j::graph::Node<T>::getGraph() {
            return _graph;
        }

        template <typename T>
        bool nd4j::graph::Node<T>::hasGraphEmbedded() {
            return _graph != nullptr;
        }

        template <typename T>
        void nd4j::graph::Node<T>::markInplace(bool reallyInplace) {
            _isInplace = reallyInplace;
        }

        template <typename T>
        OpClass nd4j::graph::Node<T>::getOpClass() {
            return _opClass;
        }

        template <typename T>
        bool nd4j::graph::Node<T>::hasBlockAttached() {
            return _block != nullptr;
        }



        template <typename T>
        bool nd4j::graph::Node<T>::isInplace() {
            return _isInplace;
        }

        template <typename T>
        bool nd4j::graph::Node<T>::isDivergencePoint() {
            if (hasCustomOp()) {
                return _customOp->getOpDescriptor()->isDivergent();
            } else
                return false;
        }

        template <typename T>
        void nd4j::graph::Node<T>::setActive(bool reallyActive) {
            _active = reallyActive;
        }

        template <typename T>
        bool nd4j::graph::Node<T>::isActive() {
            return _active;
        }

        template <typename T>
        Block<T> * nd4j::graph::Node<T>::getBlock() {
            return _block;
        }

        template <typename T>
        void nd4j::graph::Node<T>::setBlock(Block<T> *block) {
            if (_block != nullptr)
                throw "Block already exists";

            _block = block;
        }

        template <typename T>
        void nd4j::graph::Node<T>::setId(int id) {
            _id = id;
        }

        template <typename T>
        nd4j::ops::DeclarableOp<T>* nd4j::graph::Node<T>::getCustomOp() {
            return _customOp;
        }

        template <typename T>
        void nd4j::graph::Node<T>::setCustomOp(nd4j::ops::DeclarableOp<T> *customOp) {
            _customOp = customOp;
        }

        template <typename T>
        bool nd4j::graph::Node<T>::hasCustomOp() {
            return _customOp != nullptr;
        }

        template <typename T>
        std::string * nd4j::graph::Node<T>::getName() {
            return &_name;
        }

        template <typename T>
        void nd4j::graph::Node<T>::setName(const std::string& name) {
            _name = name.c_str();
        }

        template <typename T>
        void nd4j::graph::Node<T>::setName(std::string *name) {
            _name = *name;
        }

        template <typename T>
        T nd4j::graph::Node<T>::scalar() {
            return (T) _scalar;
        };

        template <typename T>
        void nd4j::graph::Node<T>::pickInput(std::pair<int,int>& pair) {
            _input.push_back(pair);
        }

        template <typename T>
        void nd4j::graph::Node<T>::pickInput(int inputId, int outputId) {
            std::pair<int,int> p(inputId,outputId);
            pickInput(p);
        }

        template <typename T>
        void nd4j::graph::Node<T>::pickInput(int inputId) {
            pickInput(inputId, 0);

            if (inputId < 0)
                _hasExternalInputs = true;
            else
                _hasInternalInputs = true;
        }

        template <typename T>
        void nd4j::graph::Node<T>::pickExternalOutput(int outputId) {
            _output.push_back(outputId);

            _hasExternalOutputs = true;
        }

        template <typename T>
        void nd4j::graph::Node<T>::pickOutput(int outputId) {
            _output.push_back(outputId);

            if (outputId < 0)
                _hasExternalOutputs = true;
            else
                _hasInternalOutputs = true;
        }

        template <typename T>
        int * nd4j::graph::Node<T>::getDimensionsPtr() {
            return _dim;
        }

        template <typename T>
        std::vector<int> * nd4j::graph::Node<T>::getDimensions() {
            return &_dimensions;
        }

        template <typename T>
        int nd4j::graph::Node<T>::getLayer() {
            return _layer;
        }

        template <typename T>
        void nd4j::graph::Node<T>::setLayer(int layer) {
            _layer = layer;
        }

        template <typename T>
        bool nd4j::graph::Node<T>::hasExternalOutputs() {
            return _hasExternalOutputs;
        }

        template <typename T>
        bool nd4j::graph::Node<T>::hasExternalInputs() {
            return _hasExternalInputs;
        }

        template <typename T>
        bool nd4j::graph::Node<T>::hasInternalOutputs() {
            return _hasInternalOutputs;
        }

        template <typename T>
        bool nd4j::graph::Node<T>::hasInternalInputs() {
            return _hasInternalInputs;
        }

        template <typename T>
        bool nd4j::graph::Node<T>::isMultiInput() {
            return _input.size() > 1;
        }

        template <typename T>
        bool nd4j::graph::Node<T>::isMultiOutput() {
            return _output.size() > 1;
        }

        template <typename T>
        T * nd4j::graph::Node<T>::extraParams() {
            return _extraParams;
        }

        template <typename T>
        nd4j::graph::OpType nd4j::graph::Node<T>::opType() {
            return _opType;
        }

        template <typename T>
        int nd4j::graph::Node<T>::id() {
            return _id;
        }

        template <typename T>
        Nd4jIndex nd4j::graph::Node<T>::opNum() {
            return _opNum;
        }

        template <typename T>
        std::vector<std::pair<int,int>> *nd4j::graph::Node<T>::input() {
            return &_input;
        }

        template <typename T>
        std::vector<int> *nd4j::graph::Node<T>::output() {
            return &_output;
        }

        template <typename T>
        nd4j::graph::Node<T>::Node(OpType opType, int opNum, int id, std::initializer_list<int> input, std::initializer_list<int> output, std::initializer_list<int> dimensions, float scalar) {
            this->_opType = opType;
            this->_id = id;
            this->_opNum = opNum;
            this->_extraParams = nullptr;
            this->_dim = nullptr;

            _hasExternalInputs = false;
            _hasExternalOutputs = false;
            _hasInternalInputs = false;
            _hasInternalOutputs = false;

            _scalar = scalar;

            for (auto i: input)
                pickInput(i);

            for (auto o: output)
                pickOutput(o);

            if (dimensions.size() > 0) {
                _dim = new int[dimensions.size()];
                int cnt = 0;
                for (auto d: dimensions) {
                    _dimensions.push_back(d);
                    _dim[cnt++] = d;
                }
            }

            // these ops allow in-place execution by design
            if (opType == OpType_TRANSFORM || opType == OpType_SCALAR || opType == OpType_BROADCAST) {
                if (_output.size() <= 1) {
                    _isInplace = true;
                }
                _opClass = OpClass_TRANSFORM;
            } else if (opType == OpType_ACCUMULATION || opType == OpType_SUMMARYSTATS) {
                _opClass = OpClass_REDUCTION;
            }
        };

        template <typename T>
        nd4j::graph::Node<T>::Node(const nd4j::graph::FlatNode *node) {
            _hasExternalInputs = false;
            _hasExternalOutputs = false;
            _hasInternalInputs = false;
            _hasInternalOutputs = false;
            _extraParams = nullptr;
            _dim = nullptr;

            _scalar = node->scalar();

            if (node != nullptr) {
                this->_id = node->id();
                this->_dataType = node->dataType();
                this->_opNum = node->opNum();
                this->_opType = node->opType();

                if (node->name() != nullptr && node->name()->c_str() != nullptr)
                    this->_name = node->name()->str();

                nd4j_verbose("Pulled node_%i (%s)\n", node->id(), this->_name.c_str())

                if (node->inputPaired() != nullptr && node->inputPaired()->size() > 0) {
                    for (int e = 0; e < (int) node->inputPaired()->size(); e++) {
                        auto pair = node->inputPaired()->Get(e);
                        pickInput(pair->first(), pair->second());
                    }
                } else if (node->input() != nullptr)
                    for (int e = 0; e < (int) node->input()->size(); e++)
                        pickInput(node->input()->Get(e));

                if (node->output() != nullptr)
                    for (int e = 0; e < (int) node->output()->size(); e++) {
                        nd4j_verbose("Picking output: %i\n", node->output()->Get(e));
                        pickOutput(node->output()->Get(e));
                    }


                if (node->extraParams() != nullptr && node->extraParams()->size() > 0) {
                    _extraParams = new T[node->extraParams()->size()];
                    for (int e = 0; e < (int) node->extraParams()->size(); e++) {
                        _extraParams[e] = (T) node->extraParams()->Get(e);
                    }
                }

                if (node->dimensions() != nullptr && node->dimensions()->size() > 0) {
                    _dim = new int[node->dimensions()->size()];
                    for (int e = 0; e < (int) node->dimensions()->size(); e++) {
                        _dimensions.push_back(node->dimensions()->Get(e));
                        _dim[e] = node->dimensions()->Get(e);
                    }
                }


                // these ops allow in-place execution by design
                if (this->_opType == OpType_TRANSFORM || this->_opType == OpType_SCALAR || this->_opType == OpType_BROADCAST) {
                    if (_output.size() <= 1) {
                        _isInplace = true;
                    }
                    _opClass = OpClass_TRANSFORM;
                } else if (this->_opType == OpType_ACCUMULATION || this->_opType == OpType_SUMMARYSTATS) {
                    _opClass = OpClass_REDUCTION;
                } else if (this->_opType == OpType_CUSTOM) {
                    auto op = nd4j::ops::OpRegistrator::getInstance()->getOperationFloat(this->opNum());
                    if (op == nullptr) {
                        nd4j_verbose("Can't find operation: %lld\n", this->opNum());
                        throw "Boom";
                    }

                    auto block = new Block<T>(this->id(), nullptr);

                    if (node->extraInteger() != nullptr)
                        for (uint32_t e = 0; e < node->extraInteger()->size(); e++)
                            block->getIArguments()->push_back(node->extraInteger()->Get(e));

                    if (node->extraParams() != nullptr)
                        for (uint32_t e = 0; e < node->extraParams()->size(); e++)
                            block->getTArguments()->push_back(node->extraParams()->Get(e));

                    this->setBlock(block);

                    this->setCustomOp(op);
                }
            } else {
                // empty dynamic node, tests probably
            }
        }

        template <typename T>
        nd4j::graph::Node<T>::~Node() {
            if (_extraParams != nullptr)
                delete[] _extraParams;

            if (_dim != nullptr)
                delete[] _dim;

            if (_block != nullptr)
                delete _block;
        }

        template <typename T>
        bool nd4j::graph::Node<T>::equals(Node *other) {
            if (_opType == other->_opType && _dataType == other->_dataType && _opNum == other->_opNum)
                return true;

            return false;
        }

        template class Node<float>;
      //  template class Node<float16>;
      //  template class Node<double>;
    }
}
