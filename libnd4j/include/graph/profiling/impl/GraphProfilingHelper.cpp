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
// Created by raver119 on 21.02.18.
//

#include <graph/profiling/GraphProfilingHelper.h>
#include <GraphExecutioner.h>

namespace nd4j {
    namespace graph {
        GraphProfile *GraphProfilingHelper::profile(Graph *graph, int iterations) {

            // saving original workspace
            auto varSpace = graph->getVariableSpace()->clone();

            // printing out graph structure
            // graph->printOut();

            // warm up
            for (int e = 0; e < 1000; e++) {
                FlowPath fp;

                auto _vs = varSpace->clone();
                //_vs->workspace()->expandTo(100000);
                _vs->setFlowPath(&fp);
                GraphExecutioner::execute(graph, _vs);

                delete _vs;
            }


            auto profile = new GraphProfile();
            for (int e = 0; e < iterations; e++) {
                FlowPath fp;

                // we're always starting from "fresh" varspace here
                auto _vs = varSpace->clone();
                //_vs->workspace()->expandTo(100000);
                _vs->setFlowPath(&fp);
                GraphExecutioner::execute(graph, _vs);

                auto p = fp.profile();
                if (e == 0)
                    profile->assign(p);
                else
                    profile->merge(p);

                delete _vs;
            }

            delete varSpace;

            return profile;
        }
    }
}
