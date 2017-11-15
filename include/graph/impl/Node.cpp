//
// @author raver119@gmail.com
//

#include <graph/Node.h>
#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/LegacyTransformOp.h>
#include <ops/declarable/LegacyScalarOp.h>
#include <ops/declarable/LegacyReduceOp.h>
#include <ops/declarable/LegacyIndexReduceOp.h>
#include <ops/declarable/LegacyStatsOp.h>
#include <ops/declarable/LegacyBroadcastOp.h>
#include <ops/declarable/LegacyReduce3Op.h>
#include <ops/declarable/LegacyPairwiseTransformOp.h>

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
            if (_block != nullptr) {
                _block->markInplace(reallyInplace);
            }
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
        Context<T> * nd4j::graph::Node<T>::getBlock() {
            return _block;
        }

        template <typename T>
        void nd4j::graph::Node<T>::setBlock(Context<T> *block) {
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

            // divergent ops (Switch etc) are always inplace, they don't allocate anything
            if (customOp->getOpDescriptor()->isDivergent())
                _isInplace = true;
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
            std::pair<int, int> pair(outputId, 0);
            _output.push_back(pair);

            _hasExternalOutputs = true;
        }

        template <typename T>
        void nd4j::graph::Node<T>::pickOutput(int outputId) {
            std::pair<int, int> pair(outputId, 0);
            _output.emplace_back(pair);

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
        std::vector<std::pair<int, int>> *nd4j::graph::Node<T>::output() {
            return &_output;
        }

        template <typename T>
        bool Node<T>::isScoped() {
            return _scope_id != 0;
        }

        template <typename T>
        void Node<T>::setScopeInfo(int id, const char* name) {
            _scope_id = id;

            if (name != nullptr)
                _scope_name = name;
        }

        template <typename T>
        int Node<T>::scopeId() {
            return _scope_id;
        }

        template <typename T>
        std::string* Node<T>::scopeName() {
            return &_scope_name;
        }

        template <typename T>
        nd4j::graph::Node<T>::Node(OpType opType, int opNum, int id, std::initializer_list<int> input, std::initializer_list<int> output, std::initializer_list<int> dimensions, float scalar, std::initializer_list<T> tArgs, std::initializer_list<int> iArgs) {
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


            if (opType == OpType_BROADCAST ||
                    opType == OpType_INDEX_ACCUMULATION ||
                    opType == OpType_SUMMARYSTATS ||
                    opType == OpType_ACCUMULATION ||
                    opType == OpType_TRANSFORM ||
                    opType == OpType_SCALAR) {

                this->setCustomOp(Node<T>::buildOpByType(opType, (int) input.size(), opNum, scalar));
                this->_isDeductable = true;

                auto block = new Context<T>(this->id(), nullptr, false);

                // there's no other IArgs in legacy options, actually
                for (auto v: dimensions)
                    block->getIArguments()->emplace_back(v);

                for (auto v: iArgs)
                    block->getIArguments()->emplace_back(v);

                for (auto v: tArgs)
                    block->getTArguments()->emplace_back(v);

                this->setBlock(block);
            } else if (opType == OpType_CUSTOM) {
                auto block = new Context<T>(this->id(), nullptr, false);

                for (auto v: iArgs)
                    block->getIArguments()->emplace_back(v);

                for (auto v: tArgs)
                    block->getTArguments()->emplace_back(v);

                this->setBlock(block);
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

            if (node->scope_id() != 0)
                this->_scope_id = node->scope_id();

            if (node->scope_name() != nullptr && node->scope_name()->size() > 0)
                this->_scope_name = node->scope_name()->str();


            _scalar = node->scalar();

            if (node != nullptr) {
                this->_id = node->id();
                this->_dataType = node->dataType();
                this->_opNum = node->opNum();
                this->_opType = node->opType();

                if (node->name() != nullptr && node->name()->c_str() != nullptr) {
                    this->_name = node->name()->str();
                }

                if (node->inputPaired() != nullptr && node->inputPaired()->size() > 0) {
                    for (int e = 0; e < (int) node->inputPaired()->size(); e++) {
                        auto pair = node->inputPaired()->Get(e);
                        pickInput(pair->first(), pair->second());
                    }
                } else if (node->input() != nullptr && node->input()->size() > 0) {
                    for (int e = 0; e < (int) node->input()->size(); e++)
                        pickInput(node->input()->Get(e));
                } else {
                    if (this->opType() != OpType_LOGIC) {
                        if (this->_name.size() > 0) {
                            nd4j_printf("Node [%i:<%s>] do not have any inputs defined\n", this->_id, this->_name.c_str());
                        } else {
                            nd4j_printf("Node [%i:<noname>] do not have any inputs defined\n", this->_id);
                        }
                    }
                }

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
                if (this->_opType == OpType_TRANSFORM || this->_opType == OpType_SCALAR || this->_opType == OpType_BROADCAST || this->_opType == OpType_ACCUMULATION || this->_opType == OpType_SUMMARYSTATS || this->_opType == OpType_INDEX_ACCUMULATION) {
                    if (_output.size() <= 1) {
                        _isInplace = true;
                    }

                    if (node->input() != nullptr && node->input()->size() > 0) {
                        this->setCustomOp(Node<T>::buildOpByType(_opType, (int) node->input()->size(), _opNum, _scalar));
                        this->_isDeductable = true;

                        auto block = new Context<T>(this->id(), nullptr, false);

                        // there's no other IArgs in legacy options, actually
                        for (auto v: _dimensions)
                            block->getIArguments()->emplace_back(v);

                        if (node->extraParams() != nullptr && node->extraParams()->size() > 0)
                            for (int e = 0; e < (int) node->extraParams()->size(); e++) {
                                block->getTArguments()->emplace_back((T) node->extraParams()->Get(e));
                            }

                        this->setBlock(block);
                    } else if (node->inputPaired() != nullptr && node->inputPaired()->size() > 0) {
                        this->setCustomOp(Node<T>::buildOpByType(_opType, (int) node->inputPaired()->size(), _opNum, _scalar));
                        this->_isDeductable = true;

                        auto block = new Context<T>(this->id(), nullptr, false);

                        for (int e = 0; e < this->input()->size(); e++) {
                            block->inputs()->emplace_back(this->input()->at(e));
                        }

                        // there's no other IArgs in legacy options, actually
                        for (auto v: _dimensions)
                            block->getIArguments()->emplace_back(v);

                        if (node->extraParams() != nullptr && node->extraParams()->size() > 0)
                            for (int e = 0; e < (int) node->extraParams()->size(); e++) {
                                block->getTArguments()->emplace_back((T) node->extraParams()->Get(e));
                            }

                        this->setBlock(block);
                    }
                } else if (this->_opType == OpType_CUSTOM) {
                    auto op = nd4j::ops::OpRegistrator::getInstance()->getOperationFloat(this->opNum());
                    if (op == nullptr) {
                        nd4j_verbose("Can't find operation: %lld\n", this->opNum());
                        throw "Boom";
                    }

                    auto block = new Context<T>(this->id(), nullptr);

                    for (int e = 0; e < this->input()->size(); e++) {
                        block->inputs()->emplace_back(this->input()->at(e));
                    }

                    if (node->extraInteger() != nullptr)
                        for (uint32_t e = 0; e < node->extraInteger()->size(); e++) {
                            int v = node->extraInteger()->Get(e);
                            block->getIArguments()->push_back(v);
                        }

                    if (node->extraParams() != nullptr)
                        for (uint32_t e = 0; e < node->extraParams()->size(); e++)
                            block->getTArguments()->emplace_back(node->extraParams()->Get(e));

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

            if (_isDeductable && _customOp != nullptr)
                delete _customOp;

        }

        template <typename T>
        bool nd4j::graph::Node<T>::equals(Node *other) {
            if (_opType == other->_opType && _dataType == other->_dataType && _opNum == other->_opNum)
                return true;

            return false;
        }

        template <typename T>
        nd4j::ops::DeclarableOp<T>* nd4j::graph::Node<T>::buildOpByType(OpType opType, int numInputs, int opNum, T scalar) {
            switch (opType) {
                case OpType_TRANSFORM:
                    if (numInputs == 2)
                        return new nd4j::ops::LegacyPairwiseTransformOp<T>(opNum);
                    else
                        return new nd4j::ops::LegacyTransformOp<T>(opNum);
                case OpType_SCALAR:
                    return new nd4j::ops::LegacyScalarOp<T>(opNum, scalar);
                case OpType_ACCUMULATION:
                    if (numInputs == 2)
                        return new nd4j::ops::LegacyReduce3Op<T>(opNum);
                    else
                        return new nd4j::ops::LegacyReduceOp<T>(opNum);
                case OpType_INDEX_ACCUMULATION:
                    return new nd4j::ops::LegacyIndexReduceOp<T>(opNum);
                case OpType_SUMMARYSTATS:
                    return new nd4j::ops::LegacyStatsOp<T>(opNum);
                case OpType_BROADCAST:
                    return new nd4j::ops::LegacyBroadcastOp<T>(opNum);
                default:
                    throw "Bad opType passed in";
            }
        }

        template class ND4J_EXPORT Node<float>;
      //  template class ND4J_EXPORT Node<float16>;
      //  template class ND4J_EXPORT Node<double>;
    }
}
