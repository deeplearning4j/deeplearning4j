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

package org.deeplearning4j.nn.conf.layers.samediff;

import lombok.Data;
import org.nd4j.base.Preconditions;

import java.util.Arrays;
import java.util.List;

/**
 * SDVertexParams is used to define the inputs - and the parameters - for a SameDiff vertex
 *
 * @author Alex Black
 */
@Data
public class SDVertexParams extends SDLayerParams {

    protected List<String> inputs;

    /**
     * Define the inputs to the DL4J SameDiff Vertex with specific names
     * @param inputNames Names of the inputs. Number here also defines the number of vertex inputs
     * @see #defineInputs(int)
     */
    public void defineInputs(String... inputNames){
        Preconditions.checkArgument(inputNames != null && inputNames.length > 0,
                "Input names must not be null, and must have length > 0: got %s", inputNames);
        this.inputs = Arrays.asList(inputNames);
    }

    /**
     * Define the inputs to the DL4J SameDiff vertex with generated names. Names will have format "input_0", "input_1", etc
     *
     * @param numInputs Number of inputs to the vertex.
     */
    public void defineInputs(int numInputs){
        Preconditions.checkArgument(numInputs > 0, "Number of inputs must be > 0: Got %s", numInputs);
        String[] inputNames = new String[numInputs];
        for( int i=0; i<numInputs; i++ ){
            inputNames[i] = "input_" + i;
        }
    }

}
