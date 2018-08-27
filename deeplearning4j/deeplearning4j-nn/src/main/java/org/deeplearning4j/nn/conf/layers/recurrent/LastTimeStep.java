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

package org.deeplearning4j.nn.conf.layers.recurrent;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.wrapper.BaseWrapperLayer;
import org.deeplearning4j.nn.layers.recurrent.LastTimeStepLayer;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;

/**
 * LastTimeStep is a "wrapper" layer: it wraps any RNN (or CNN1D) layer, and extracts out the last time step during forward pass,
 * and returns it as a row vector (per example). That is, for 3d (time series) input (with shape [minibatch, layerSize,
 * timeSeriesLength]), we take the last time step and return it as a 2d array with shape [minibatch, layerSize].<br>
 * Note that the last time step operation takes into account any mask arrays, if present: thus, variable length time
 * series (in the same minibatch) are handled as expected here.
 *
 * @author Alex Black
 */
public class LastTimeStep extends BaseWrapperLayer {

    private LastTimeStep(){ }

    public LastTimeStep(Layer underlying){
        super(underlying);
        this.layerName = underlying.getLayerName(); // needed for keras import to match names
    }

    public Layer getUnderlying() {
        return underlying;
    }


    @Override
    public org.deeplearning4j.nn.api.Layer instantiate(NeuralNetConfiguration conf, Collection<TrainingListener> trainingListeners,
                                                       int layerIndex, INDArray layerParamsView, boolean initializeParams) {
        NeuralNetConfiguration conf2 = conf.clone();
        conf2.setLayer(((LastTimeStep)conf2.getLayer()).getUnderlying());
        return new LastTimeStepLayer(underlying.instantiate(conf2, trainingListeners, layerIndex, layerParamsView, initializeParams));
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        if(inputType.getType() != InputType.Type.RNN){
            throw new IllegalArgumentException("Require RNN input type - got " + inputType);
        }
        InputType outType = underlying.getOutputType(layerIndex, inputType);
        InputType.InputTypeRecurrent r = (InputType.InputTypeRecurrent)outType;
        return InputType.feedForward(r.getSize());
    }
}
