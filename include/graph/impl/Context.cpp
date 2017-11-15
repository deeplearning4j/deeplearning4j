//
// @author raver119@gmail.com
//

#include <Context.h>

namespace nd4j {
    namespace graph {

        template <typename T>
        Context<T>::Context(int nodeId, VariableSpace<T> *variableSpace) {
            _nodeId = nodeId;
            _variableSpace = variableSpace;
            _isInplace = false;
            _workspace = nullptr;

            _executionTime.first = 0;
            _executionTime.second = 0;
        }

        template <typename T>
        Context<T>::Context(int nodeId, VariableSpace<T> *variableSpace, bool isInplace) : Context(nodeId, variableSpace) {
            _isInplace = isInplace;
        }

        template<typename T>
        Context<T>::~Context() {
            _iArgs.clear();
            _tArgs.clear();
            _inputs.clear();
        }


        template <typename T>
        bool Context<T>::hasWorkspaceProvided() {
            return _workspace != nullptr;
        }

        template <typename T>
        void Context<T>::markInplace(bool reallyInplace) {
            _isInplace = reallyInplace;
        }

        template <typename T>
        void Context<T>::attachWorkspace(nd4j::memory::Workspace* workspace) {
            _workspace = workspace;
        }

        template <typename T>
        void Context<T>::setVariableSpace(VariableSpace<T> *variableSpace) {
            this->_variableSpace = variableSpace;
        }

        template <typename T>
        void Context<T>::forgetWorkspace() {
            _workspace = nullptr;
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

        template <typename T>
        int Context<T>::nodeId() {
            return getNodeId();
        }

        template <typename T>
        int Context<T>::getNodeId() {
            return _nodeId;
        }

        /**
         * This method returns number of inputs available in this block
         * @return
         */
        template <typename T>
        unsigned long Context<T>::width() {
            return _inputs.size();
        };

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

        template <typename T>
        bool Context<T>::isInplace() {
            return _isInplace;
        }

        template <typename T>
        std::vector<T>* Context<T>::getTArguments() {
            return &_tArgs;
        }

        template <typename T>
        std::vector<int>* Context<T>::getIArguments() {
            return &_iArgs;
        }

        template <typename T>
        void Context<T>::pickInput(int input) {
            std::pair<int, int> pair(input, 0);
            _inputs.emplace_back(pair);
        }

        template <typename T>
        std::pair<int, int>* Context<T>::input(int idx) {
            return &(_inputs.at(idx));
        }

        template <typename T>
        void Context<T>::fillInputs(std::initializer_list<int> inputs) {
            for (auto v: inputs) {
                pickInput(v);
            }
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
            return _branch;
        }

        template <typename T>
        void Context<T>::setBranch(int branch) {
            _branch = branch;
        }

        template <typename T>
        void Context<T>::fillInputs(std::vector<int>& inputs) {
            for (int e = 0; e < inputs.size(); e++) {
                auto v = inputs.at(e);
                pickInput(v);
            }
        }

        template <typename T>
        Nd4jIndex nd4j::graph::Context<T>::getOuterTime(){
            return _executionTime.first;
        }

        template <typename T>
        Nd4jIndex nd4j::graph::Context<T>::getInnerTime(){
            return _executionTime.second;
        }

        template <typename T>
        std::vector<std::pair<int, int>>* nd4j::graph::Context<T>::inputs() {
            return &_inputs;
        }

        template <typename T>
        void nd4j::graph::Context<T>::setOuterTime(Nd4jIndex time){
            _executionTime.first = time;
        }

        template <typename T>
        void nd4j::graph::Context<T>::setInnerTime(Nd4jIndex time){
            _executionTime.second = time;
        }

        template <typename T>
        bool nd4j::graph::Context<T>::hasVariablesFilled() {
            return _inputs.size() > 0;
        }


        template <typename T>
        int Context<T>::opNum() {
            return _opNum;
        }

        template <typename T>
        void Context<T>::setOpNum(int opNum) {
            _opNum = opNum;
        }

        template <typename T>
        Variable<T>* Context<T>::getVariable(int idx) {
            if (idx >= _inputs.size()) {
                nd4j_printf("Node %i; Variable [%i] requested, but only %i inputs available\n", _nodeId, idx, _inputs.size());
                throw "Bad index";
            }

            auto p = _inputs[idx];
            return variable(p);
        }

        template <typename T>
        Variable<T>* Context<T>::variable(int idx) {
            return getVariable(idx);
        }

        template <typename T>
        Variable<T>* Context<T>::variable(std::initializer_list<int> p) {
            if (p.size() != 2)
                throw "Variable address should have size of 2";

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
                nd4j_printf("Node %i; Non-existent variable requested: [%i:%i]\n", _nodeId, p.first, p.second);
                throw "Bad variable";
            }

            return _variableSpace->getVariable(p);
        }

        template <typename T>
        void Context<T>::pickInput(std::pair<int, int>& p) {
            _inputs.emplace_back(p);
        }

        template <typename T>
        void Context<T>::pickInput(int input, int index) {
            std::pair<int, int> pair(input, index);
            pickInput(pair);
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
            std::pair<int, int> pair(nodeId(), idx);
            if (!_variableSpace->hasVariable(pair)) {
                auto var = new Variable<T>(nullptr, nullptr, nodeId(), idx);
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


        template class ND4J_EXPORT Context<float>;
        template class ND4J_EXPORT Context<float16>;
        template class ND4J_EXPORT Context<double>;
    }
}

