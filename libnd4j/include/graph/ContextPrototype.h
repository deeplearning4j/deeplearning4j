/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
//  @author raver119@gmail.com
//

#ifndef ND4J_CONTEXT_PROTOTYPE_H
#define ND4J_CONTEXT_PROTOTYPE_H

#include <vector>
#include "RandomGenerator.h"

namespace nd4j {
    namespace graph {
        template <typename T>
        class ContextPrototype {
        protected:
            // int ids of the input nodes
            std::vector<std::pair<int, int>> _inputs;
            int _nodeId;
            std::vector<T> _tArgs;
            std::vector<int> _iArgs;            
			
			bool _isInplace;

            // opNum for legacy XYZ ops
            int _opNum = -1;
            uint64_t _rootSeed;
            RandomGenerator _randomGenerator;

            bool _useMKLDNN = true;

        public:
            explicit ContextPrototype(int nodeId = 1, bool inPlace = false);
            ~ContextPrototype() = default;

            int getNodeId();
            int nodeId();

            // this method returns true, if inputs are defined
            bool hasVariablesFilled();

            bool isInplace();
            void markInplace(bool reallyInplace);

            void pickInput(int input);
            void pickInput(int input, int index);
            void pickInput(std::pair<int, int>& p);
            void fillInputs(std::initializer_list<int> inputs);
            void fillInputs(std::vector<int>& inputs);
            std::vector<std::pair<int, int>>* inputs();

            std::vector<T>* getTArguments();
            std::vector<int>* getIArguments();

            int numT();
            int numI();

            std::pair<int, int>* input(int idx);

            int opNum();
            void setOpNum(int opNum);

            bool isUseMKLDNN() { return _useMKLDNN; }
            void setUseMKLDNN(bool useMKLDNN) { _useMKLDNN = useMKLDNN; }

            /**
             * This method returns number of inputs available in this block
             * @return
             */
            unsigned long width();

            // just a clone
            ContextPrototype<T>* clone();

            template <typename N>
            ContextPrototype<N>* asT();

            RandomGenerator& randomGenerator() {return _randomGenerator;}
            RandomGenerator const& getRng() const { return _randomGenerator; }
            uint64_t randomSeed() const { return _rootSeed; }
            void setRandomSeed(uint64_t seed) { _rootSeed = seed; }
        };
    }
}

#endif //ND4J_CONTEXT_PROTOTYPE_H