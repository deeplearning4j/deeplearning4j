/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
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

/**
 * This op calculates gains - data used internally by Barnes-Hut-TSNE algorithm.
 *
 * @author alexander.stoyakin@gmail.com
 */
public class BarnesHutGains extends DynamicCustomOp {

    public BarnesHutGains(){ }

    public BarnesHutGains(INDArray output, INDArray input, INDArray gradx, INDArray epsilon) {

        inputArguments.add(input);
        inputArguments.add(gradx);
        inputArguments.add(epsilon);

        outputArguments.add(output);
    }

    @Override
    public String opName() {
        return "barnes_gains";
    }
}
