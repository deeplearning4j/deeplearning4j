/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.autodiff.samediff.transform;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;

import java.util.List;

/**
 * SubGraphProcessor is used in {@link GraphTransformUtil} to define how a subgraph should be modified or replaced.
 * Note that when replacing a subgraph, the replacement subgraph must have the same number of outputs as the original subgraph.
 * Note that the order of the outputs matter.
 *
 * @author Alex Black
 */
public interface SubGraphProcessor {

    /**
     * Replace the subgraph, and return the new outputs that should replace the old outputs.<br>
     * Note that the order of the outputs you return matters!<br>
     * If the original outputs are [A,B,C] and you return output variables [X,Y,Z], then anywhere "A" was used as input
     * will now use "X"; similarly Y replaces B, and Z replaces C.
     *
     * @param sd SameDiff instance
     * @param subGraph Subgraph to modify
     * @return New output variables
     */
    List<SDVariable> processSubgraph(SameDiff sd, SubGraph subGraph);

}
