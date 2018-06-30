//
// @author raver119@gmail.com
//

#include <Context.h>
#include <helpers/ShapeUtils.h>

namespace nd4j {
    namespace graph {

        template <typename T>
        Context<T>::Context(ContextPrototype<T>* prototype, VariableSpace<T>* variableSpace) {
            _variableSpace = variableSpace;

            if (prototype != nullptr) {
                for (const auto &v: *(prototype->inputs())) {
                    this->_inputs.push_back(v);
                }

                for (const auto &v: *(prototype->getTArguments())) {
                    this->_tArgs.push_back(v);
                }

                for (const auto &v: *(prototype->getIArguments())) {
                    this->_iArgs.push_back(v);
                }

                this->_opNum = prototype->opNum();
                this->_isInplace = prototype->isInplace();
                this->_nodeId = prototype->nodeId();
            }


            if (variableSpace != nullptr && variableSpace->workspace() != nullptr)
                    this->_workspace = variableSpace->workspace();
        }


        template <typename T>
        Context<T>::Context(int nodeId, VariableSpace<T> *variableSpace) {
            this->_nodeId = nodeId;
            this->_variableSpace = variableSpace;
            this->_isInplace = false;
            this->_workspace = nullptr;

            this->_executionTime.first = 0;
            this->_executionTime.second = 0;

            if (variableSpace != nullptr)
                this->_rng = variableSpace->getRNG();

            if (variableSpace != nullptr && variableSpace->workspace() != nullptr)
                this->_workspace = variableSpace->workspace();
        }

        template <typename T>
        Context<T>::Context(int nodeId, VariableSpace<T> *variableSpace, bool isInplace) : Context<T>(nodeId, variableSpace) {
            this->_isInplace = isInplace;
        }

        template<typename T>
        Context<T>::~Context() {
            this->_iArgs.clear();
            this->_tArgs.clear();
            this->_inputs.clear();
        }

        template <typename T>
        bool Context<T>::hasWorkspaceProvided() {
            return this->_workspace != nullptr;
        }

        template <typename T>
        void Context<T>::attachWorkspace(nd4j::memory::Workspace* workspace) {
            this->_workspace = workspace;
        }

        template <typename T>
        void Context<T>::setVariableSpace(VariableSpace<T> *variableSpace) {
            this->_variableSpace = variableSpace;

            if (variableSpace != nullptr)
                this->_rng = variableSpace->getRNG();
        }

        template <typename T>
        void Context<T>::forgetWorkspace() {
            _workspace = nullptr;
        }

        template<typename T>
        VariableSpace<T> *Context<T>::getVariableSpace() {
            return _variableSpace;
        }

        template <typename T>
        nd4j::memory::Workspace* Context<T>::getWorkspace() {
            return _workspace;
        }

        template <typename T>
        nd4j::memory::Workspace* Context<T>::workspace() {
            return _workspace;
        }

        template <typename T>
        nd4j::random::RandomBuffer* Context<T>::getRNG() {
            return _rng;
        }

        template <typename T>
        void Context<T>::setRNG(nd4j::random::RandomBuffer* rng) {
            _rng = rng;
        }

        /**
         * This method returns variableSpace used in this block
         * @return
         */
    /*    template <typename T>
        VariableSpace<T>* Context<T>::getVariableSpace() {
            return _variableSpace;
        }
*/

        template <typename T>
        Stash<T>* Context<T>::getStash() {
            return _variableSpace->getStash();
        }

        template <typename T>
        void Context<T>::trackList(NDArrayList<T>* list) {
            _variableSpace->trackList(list);
        }

/*
        template <typename T>
        void Block<T>::updateVariables() {
            _variables.clear();
            auto x = _inputs.size();
            for (auto &v:_inputs) {
                auto var = _variableSpace->getVariable(v);
                _variables.emplace_back(var);
            }
        }
*/
        template <typename T>
        int Context<T>::getBranch() {
            return _variableSpace->flowPath()->branch(this->nodeId());
        }

        template <typename T>
        void Context<T>::setBranch(int branch) {
            //_branch = branch;
            if (_variableSpace->flowPath() != nullptr)
                _variableSpace->flowPath()->markBranch(this->nodeId(), branch);
        }

        template <typename T>
        Nd4jLong nd4j::graph::Context<T>::getOuterTime(){
            return this->_executionTime.first;
        }

        template <typename T>
        Nd4jLong nd4j::graph::Context<T>::getInnerTime(){
            return this->_executionTime.second;
        }

        template <typename T>
        void nd4j::graph::Context<T>::setOuterTime(Nd4jLong time){
            this->_executionTime.first = time;
        }

        template <typename T>
        void nd4j::graph::Context<T>::setInnerTime(Nd4jLong time){
            this->_executionTime.second = time;
        }


        template <typename T>
        Variable<T>* Context<T>::getVariable(int idx) {
            if (idx >= this->_inputs.size()) {
                nd4j_printf("Node %i; Variable [%i] requested, but only %i inputs available\n", this->_nodeId, idx, this->_inputs.size());
                throw std::runtime_error("Bad index");
            }

            auto p = this->_inputs[idx];

            auto v = variable(p);

            if (Environment::getInstance()->isDebugAndVerbose() && v != nullptr &&  v->getNDArray() != nullptr) {
                auto array = v->getNDArray();
                std::string shape_ = ShapeUtils<T>::shapeAsString(array);

                float m = std::numeric_limits<float>::quiet_NaN();
                if (!array->isEmpty()) {
                    auto values = array->asIndexedString(16);
                    nd4j_printf("Debug info for node_%i input[%i]; shape: %s; ews: %i; order: %i; first values: %s\n", this->_nodeId, idx, shape_.c_str(), array->ews(), array->ordering(), values.c_str());
                } else {
                    nd4j_printf("Debug info for node_%i input[%i]; shape: %s; ews: %i; order: %i; mean value: [%f]\n", this->_nodeId, idx, shape_.c_str(), array->ews(), array->ordering(), m);
                }
            }

            return v;
        }

        template <typename T>
        Variable<T>* Context<T>::variable(int idx) {
            return getVariable(idx);
        }

        template <typename T>
        Variable<T>* Context<T>::variable(std::initializer_list<int> p) {
            if (p.size() != 2)
                throw std::runtime_error("Variable address should have size of 2");

            // FIXME: lol
            std::vector<int> vec(p);
            std::pair<int, int> pair(vec[0], vec[1]);
            return variable(pair);
        }

        template <typename T>
        Variable<T>* Context<T>::variable(int node, int idx) {
            std::pair<int, int> pair(node, idx);
            return variable(pair);
        }

        template <typename T>
        Variable<T>* Context<T>::variable(std::pair<int,int>& p) {
            if (!_variableSpace->hasVariable(p)) {
                nd4j_printf("Node %i; Non-existent variable requested: [%i:%i]\n", this->_nodeId, p.first, p.second);
                throw std::runtime_error("Bad variable");
            }

            return _variableSpace->getVariable(p);
        }


        template <typename T>
        void Context<T>::pushNDArrayToVariableSpace(int nodeId, int index, NDArray<T> *array, bool removable) {
            std::pair<int,int> pair(nodeId, index);
            pushNDArrayToVariableSpace(pair, array, removable);
        }

        template <typename T>
        void Context<T>::pushNDArrayToVariableSpace(std::pair<int, int> &pair, NDArray<T> *array, bool removable) {
            if (!_variableSpace->hasVariable(pair)) {
                auto var = new Variable<T>(array, nullptr, pair.first, pair.second);
                _variableSpace->putVariable(pair, var);
                var->markRemovable(removable);
            } else {
                auto var = _variableSpace->getVariable(pair);
                if (var->getNDArray() != array) {
                    if (var->isRemovable() && var->getNDArray() != nullptr)
                        delete var->getNDArray();

                    var->setNDArray(array);
                    var->markRemovable(removable);
                }
            }
        }

        template <typename T>
        void Context<T>::pushNDArrayListToVariableSpace(int nodeId, int index, NDArrayList<T>* list, bool track) {
            std::pair<int,int> pair(nodeId, index);
            pushNDArrayListToVariableSpace(pair, list, track);
        }
        
        template <typename T>
        void Context<T>::pushNDArrayListToVariableSpace(std::pair<int, int>& pair, NDArrayList<T>* list, bool track) {
            if (!_variableSpace->hasVariable(pair)) {
                auto var = new Variable<T>(nullptr, nullptr, pair.first, pair.second);
                var->setNDArrayList(list);
                _variableSpace->putVariable(pair, var);
            } else {
                auto var = _variableSpace->getVariable(pair);
                var->setNDArrayList(list);
            }

            if (track)
                _variableSpace->trackList(list);
        }

        template <typename T>
        Variable<T>* Context<T>::ensureVariable(int idx) {
            std::pair<int, int> pair(this->nodeId(), idx);
            if (!_variableSpace->hasVariable(pair)) {
                auto var = new Variable<T>(nullptr, nullptr, this->nodeId(), idx);
                _variableSpace->putVariable(pair, var);
                return var;
            } else {
                return _variableSpace->getVariable(pair);
            }
        }


        template <typename T>
        bool Context<T>::isValueAvailable(int idx) {
            auto var = ensureVariable(idx);

            if (var->variableType() == VariableType::NDARRAY) {
                return var->getNDArray() != nullptr;
            } else if (var->variableType() == VariableType::ARRAY_LIST) {
                return var->getNDArrayList() != nullptr;
            }

            return false;
        }

        template<typename T>
        nd4j::memory::Workspace *Context<T>::fWorkspace() {
            return workspace();
        }

        template<typename T>
        nd4j::memory::Workspace *Context<T>::tWorkspace() {
            return nullptr;
        }

        template<typename T>
        nd4j::memory::Workspace *Context<T>::oWorkspace() {
            return nullptr;
        }

        template class ND4J_EXPORT Context<float>;
        template class ND4J_EXPORT Context<float16>;
        template class ND4J_EXPORT Context<double>;
    }
}

