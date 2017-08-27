//
// @author raver119@gmail.com
//

#ifndef LIBND4J_BLOCK_H
#define LIBND4J_BLOCK_H

#include <vector>
#include "Variable.h"
#include "VariableSpace.h"

namespace nd4j {
    namespace graph {
        /**
         * This class defines input desired for any given node/operation within graph
         */
        template <typename T>
        class Block {
        protected:
            // int ids of the input nodes
            std::vector<int> _inputs;
            std::vector<nd4j::graph::Variable<T> *> _variables;
            nd4j::graph::VariableSpace<T> * _variableSpace;

        public:
            Block(VariableSpace<T> *variableSpace) {
                _variableSpace = variableSpace;
            }

            ~Block() {
                //
            }

            /**
             * This method returns number of inputs available in this block
             * @return
             */
            unsigned long width() {
                return _inputs.size();
            };

            /**
             * This method returns variableSpace used in this block
             * @return
             */
            VariableSpace<T> *getVariableSpace() {
                return _variableSpace;
            }

            void fillInputs(std::initializer_list<int> inputs) {
                for (auto v: inputs) {
                    _inputs.push_back(v);

                    if (!_variableSpace->hasVariable(v))
                        throw "Unknown variable was referenced";

                    _variables.push_back(_variableSpace->getVariable(v));
                }
            }

            void fillInputs(std::vector<int> *inputs) {
                for (int e = 0; e < inputs->size(); e++) {
                    auto v = inputs->at(e);
                    _inputs.push_back(v);

                    if (!_variableSpace->hasVariable(v))
                        throw "Unknown variable was referenced";

                    _variables.push_back(_variableSpace->getVariable(v));
                }
            }

            std::vector<nd4j::graph::Variable<T> *>& getVariables();
        };
    }
}


/**
* This method returns variables in this block
* @return
*/
template <typename T>
std::vector<nd4j::graph::Variable<T> *>& nd4j::graph::Block<T>::getVariables() {
    return _variables;
}

#endif //LIBND4J_BLOCK_H
