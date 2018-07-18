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

package org.deeplearning4j.nn.layers.recurrent;

import org.deeplearning4j.nn.api.layers.RecurrentLayer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public abstract class BaseRecurrentLayer<LayerConfT extends org.deeplearning4j.nn.conf.layers.BaseLayer>
                extends BaseLayer<LayerConfT> implements RecurrentLayer {

    /**
     * stateMap stores the INDArrays needed to do rnnTimeStep() forward pass.
     */
    protected Map<String, INDArray> stateMap = new ConcurrentHashMap<>();

    /**
     * State map for use specifically in truncated BPTT training. Whereas stateMap contains the
     * state from which forward pass is initialized, the tBpttStateMap contains the state at the
     * end of the last truncated bptt
     */
    protected Map<String, INDArray> tBpttStateMap = new ConcurrentHashMap<>();

    public BaseRecurrentLayer(NeuralNetConfiguration conf) {
        super(conf);
    }

    public BaseRecurrentLayer(NeuralNetConfiguration conf, INDArray input) {
        super(conf, input);
    }

    /**
     * Returns a shallow copy of the stateMap
     */
    @Override
    public Map<String, INDArray> rnnGetPreviousState() {
        return new HashMap<>(stateMap);
    }

    /**
     * Set the state map. Values set using this method will be used
     * in next call to rnnTimeStep()
     */
    @Override
    public void rnnSetPreviousState(Map<String, INDArray> stateMap) {
        this.stateMap.clear();
        this.stateMap.putAll(stateMap);
    }

    /**
     * Reset/clear the stateMap for rnnTimeStep() and tBpttStateMap for rnnActivateUsingStoredState()
     */
    @Override
    public void rnnClearPreviousState() {
        stateMap.clear();
        tBpttStateMap.clear();
    }

    @Override
    public Map<String, INDArray> rnnGetTBPTTState() {
        return new HashMap<>(tBpttStateMap);
    }

    @Override
    public void rnnSetTBPTTState(Map<String, INDArray> state) {
        tBpttStateMap.clear();
        tBpttStateMap.putAll(state);
    }

}
