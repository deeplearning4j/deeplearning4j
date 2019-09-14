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
// @author raver119@gmail.com
//

#ifndef SAMEDIFF_SAMEDIFF_H
#define SAMEDIFF_SAMEDIFF_H

#include <NDArray.h>
#include "Variable.h"
#include <graph/Graph.h>
#include <unordered_map>

namespace samediff {

    class ND4J_EXPORT SameDiff {
    protected:
        // TODO: use shared_ptr here
        nd4j::graph::Graph *_graph = nullptr;
    public:
        SameDiff();
        ~SameDiff();


        Variable variable(const nd4j::NDArray &array, bool trainable = true, const std::string &name = {});
        Variable placeholder(const std::string &name, const nd4j::DataType dataType = nd4j::DataType::FLOAT32, const std::vector<Nd4jLong> shape = {-1});


        nd4j::graph::Graph* graph();

        // execution functions
        void execute();
        void executeWithDictionary(const std::unordered_map<const char*, nd4j::NDArray> &args);

        // TODO: we need to pass data in some cross-platform format. i.e. Queue. or Iterator.
        void train();

        // file operations with graphs
        void save(const char *filename);
        static SameDiff load(const char *filename);
    };

}

#endif //SAMEDIFF_SAMEDIFF_H
