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
// Created by raver119 on 14.10.2017.
//

#ifndef LIBND4J_SCOPE_H
#define LIBND4J_SCOPE_H

#include <string>
#include <map>
#include <graph/Node.h>

namespace nd4j {
    namespace graph {

        /**
         * Scope holds sequential list of operations, and made suitable for continuous
         * re-execution of multiple operations.
         *
         * @tparam T
         */
        class ND4J_EXPORT Scope {
        protected:
            // Graph-unique IDs for Scope instances
            int _id;
            std::string _name;

            // list of nodes to run, always sequential
            // Graph takes care of topo sort
            std::vector<Node*> _nodes;
        public:
            // attach GiG here, with shared namespace?
            // or just rebuilt graph leaf?
            //          ¯\_(ツ)_/¯

            // default consructor
            explicit Scope(int id, const char* name = nullptr);

            // default destructor
            ~Scope();

            /**
             * this method adds op node to the scope
             *
             * PLEASE NOTE: We assume that ops are being added ORDERED
             */
            void push_back(Node* node);

            /**
             * This method returns list of ops stored earlier, ready for execution
             *
             * PLEASE NOTE: If the scope is conditional - last op in list should be BooleanOp
             * @return
             */
            std::vector<Node*> * nodes();

            /**
             * This function returns number of nodes in this scope
             *
             * @return
             */
            int size();

            /**
             * Returns ID of this scope
             * @return
             */
            int id();

            /**
             * Returns name of this scope
             *
             * @return
             */
            std::string* name();

            /**
             * This method returns clone of this Scope
             */
            Scope* clone();

            /**
             * This method removes all Nodes from this scope
             */
            void forgetNodes();
        };
    }
}


#endif //LIBND4J_SCOPE_H
