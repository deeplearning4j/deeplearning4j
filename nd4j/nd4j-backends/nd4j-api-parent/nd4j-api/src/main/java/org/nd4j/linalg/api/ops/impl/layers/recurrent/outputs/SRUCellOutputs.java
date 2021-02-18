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

package org.nd4j.linalg.api.ops.impl.layers.recurrent.outputs;

import java.util.Arrays;
import java.util.List;
import lombok.Getter;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.GRUCell;

@Getter
public class SRUCellOutputs {


    /**
     * Current cell output [batchSize, numUnits].
     */
    private SDVariable h;

    /**
     * Current cell state [batchSize, numUnits].
     */
    private SDVariable c;

    public SRUCellOutputs(SDVariable[] outputs){
        Preconditions.checkArgument(outputs.length == 2,
                "Must have 2 SRU cell outputs, got %s", outputs.length);

        h = outputs[0];
        c = outputs[1];
    }

    /**
     * Get all outputs returned by the cell.
     */
    public List<SDVariable> getAllOutputs(){
        return Arrays.asList(h, c);
    }

    /**
     * Get h, the output of the cell.
     *
     * Has shape [batchSize, inSize].
     */
    public SDVariable getOutput(){
        return h;
    }

    /**
     * Get c, the state of the cell.
     *
     * Has shape [batchSize, inSize].
     */
    public SDVariable getState(){
        return c;
    }

}
