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

package org.nd4j.linalg.api.ops.custom;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

public class BarnesEdgeForces extends DynamicCustomOp {

    public BarnesEdgeForces(){ }

    public BarnesEdgeForces(INDArray rowP, INDArray colP, INDArray valP, INDArray dataP, long N,
                            INDArray output) {

        inputArguments.add(rowP);
        inputArguments.add(colP);
        inputArguments.add(valP);
        inputArguments.add(dataP);

        iArguments.add(N);

        outputArguments.add(output);
    }

    @Override
    public String opName() {
        return "barnes_edge_forces";
    }
}
