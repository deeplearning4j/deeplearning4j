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
// Created by GS <sgazeos@gmail.com> 3/7/2018
//

#ifndef __H__GRAPH_UTILS__
#define __H__GRAPH_UTILS__

#include <vector>
#include <ops/declarable/OpDescriptor.h>
#include <ops/declarable/DeclarableOp.h>

namespace nd4j {
namespace graph {

class ND4J_EXPORT GraphUtils {
public:
    typedef std::vector<OpDescriptor> OpList;

public:
    static bool filterOperations(OpList& ops);
    static std::string makeCommandLine(OpList& ops);
    static int runPreprocessor(char const* input, char const* output);
};

}
}
#endif
