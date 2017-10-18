//
// @author raver119@gmail.com
//

#include <Block.h>

namespace nd4j {
    namespace graph {

        template <typename T>
        Block<T>::Block(int nodeId, VariableSpace<T> *variableSpace) {
            _nodeId = nodeId;
            _variableSpace = variableSpace;
            _isInplace = false;
            _workspace = nullptr;

            _executionTime.first = 0;
            _executionTime.second = 0;
        }

        template <typename T>
        Block<T>::Block(int nodeId, VariableSpace<T> *variableSpace, bool isInplace) : Block(nodeId, variableSpace) {
            _isInplace = isInplace;
        }

        template<typename T>
        Block<T>::~Block() {
            //_variables.clear();
            //_inputs.clear();
        }


        template <typename T>
        bool Block<T>::hasWorkspaceProvided() {
            return _workspace != nullptr;
        }


        template <typename T>
        void Block<T>::attachWorkspace(nd4j::memory::Workspace* workspace) {
            _workspace = workspace;
        }

        template <typename T>
        void Block<T>::setVariableSpace(VariableSpace<T> *variableSpace) {
            this->_variableSpace = variableSpace;
        }

        template <typename T>
        void Block<T>::forgetWorkspace() {
            _workspace = nullptr;
        }

        template <typename T>
        nd4j::memory::Workspace* Block<T>::getWorkspace() {
            return _workspace;
        }

        template <typename T>
        nd4j::random::RandomBuffer* Block<T>::getRNG() {
            return _rng;
        }

        template <typename T>
        void Block<T>::setRNG(nd4j::random::RandomBuffer* rng) {
            _rng = rng;
        }

        template <typename T>
        int Block<T>::getNodeId() {
            return _nodeId;
        }

        /**
         * This method returns number of inputs available in this block
         * @return
         */
        template <typename T>
        unsigned long Block<T>::width() {
            return _inputs.size();
        };

        /**
         * This method returns variableSpace used in this block
         * @return
         */
        template <typename T>
        VariableSpace<T>* Block<T>::getVariableSpace() {
            return _variableSpace;
        }

        template <typename T>
        bool Block<T>::isInplace() {
            return _isInplace;
        }

        template <typename T>
        std::vector<T>* Block<T>::getTArguments() {
            return &_tArgs;
        }

        template <typename T>
        std::vector<int>* Block<T>::getIArguments() {
            return &_iArgs;
        }

        template <typename T>
        void Block<T>::pickInput(int input) {
            _inputs.emplace_back(input);

            if (!_variableSpace->hasVariable(input))
                throw "Unknown variable was referenced";

            _variables.emplace_back(_variableSpace->getVariable(input));
        }

        template <typename T>
        void Block<T>::fillInputs(std::initializer_list<int> inputs) {
            for (auto v: inputs) {
                pickInput(v);
            }
        }

        template <typename T>
        void Block<T>::fillInputs(std::vector<int>& inputs) {
            for (int e = 0; e < inputs.size(); e++) {
                auto v = inputs.at(e);
                pickInput(v);
            }
        }

        template <typename T>
        Nd4jIndex nd4j::graph::Block<T>::getOuterTime(){
            return _executionTime.first;
        }

        template <typename T>
        Nd4jIndex nd4j::graph::Block<T>::getInnerTime(){
            return _executionTime.second;
        }

        template <typename T>
        void nd4j::graph::Block<T>::setOuterTime(Nd4jIndex time){
            _executionTime.first = time;
        }

        template <typename T>
        void nd4j::graph::Block<T>::setInnerTime(Nd4jIndex time){
            _executionTime.second = time;
        }

        template <typename T>
        bool nd4j::graph::Block<T>::hasVariablesFilled() {
            return _variables.size() > 0;
        }


        /**
        * This method returns variables in this block
        * @return
        */
        template <typename T>
        std::vector<nd4j::graph::Variable<T> *>& nd4j::graph::Block<T>::getVariables() {
            return _variables;
        }


        template class ND4J_EXPORT Block<float>;
        template class ND4J_EXPORT Block<float16>;
        template class ND4J_EXPORT Block<double>;
    }
}

