/*
 *  ******************************************************************************
 *  *
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
package org.deeplearning4j.rl4j.network;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.HashMap;

/**
 * A class containing the output(s) of a neural net. The outputs are stored as keys-values.
 */
public class NeuralNetOutput {
    private final HashMap<String, INDArray> outputs = new HashMap<String, INDArray>();

    /**
     * Store an output with a given key
     * @param key The name of the output
     * @param output The output
     */
    public void put(String key, INDArray output) {
        outputs.put(key, output);
    }

    /**
     * @param key The name of the output
     * @return The output associated with the key
     */
    public INDArray get(String key) {
        INDArray result = outputs.get(key);
        if(result == null) {
            throw new IllegalArgumentException(String.format("There is no element with key '%s' in the neural net output.", key));
        }
        return result;
    }
}
