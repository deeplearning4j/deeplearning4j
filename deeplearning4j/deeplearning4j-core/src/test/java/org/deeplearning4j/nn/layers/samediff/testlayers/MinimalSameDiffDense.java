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

package org.deeplearning4j.nn.layers.samediff.testlayers;

import lombok.Data;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.samediff.SameDiffLayer;
import org.deeplearning4j.nn.conf.layers.samediff.SDLayerParams;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collections;
import java.util.Map;

@Data
public class MinimalSameDiffDense extends SameDiffLayer {

    private int nIn;
    private int nOut;
    private Activation activation;

    public MinimalSameDiffDense(int nIn, int nOut, Activation activation, WeightInit weightInit){
        this.nIn = nIn;
        this.nOut = nOut;
        this.activation = activation;
        this.weightInit = weightInit;
    }

    protected MinimalSameDiffDense(){
        //For JSON serialization
    }

    @Override
    public SDVariable defineLayer(SameDiff sd, SDVariable layerInput, Map<String, SDVariable> paramTable) {
        SDVariable weights = paramTable.get(DefaultParamInitializer.WEIGHT_KEY);
        SDVariable bias = paramTable.get(DefaultParamInitializer.BIAS_KEY);

        SDVariable mmul = sd.mmul("mmul", layerInput, weights);
        SDVariable z = mmul.add("z", bias);
        return activation.asSameDiff("out", sd, z);
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        return InputType.feedForward(nOut);
    }

    @Override
    public void defineParameters(SDLayerParams params) {
        params.addWeightParam(DefaultParamInitializer.WEIGHT_KEY, nIn, nOut);
        params.addBiasParam(DefaultParamInitializer.BIAS_KEY, 1, nOut);
    }

    @Override
    public void initializeParameters(Map<String, INDArray> params) {
        params.get(DefaultParamInitializer.BIAS_KEY).assign(0);
        initWeights(nIn, nOut, weightInit, params.get(DefaultParamInitializer.WEIGHT_KEY));
    }

    //OPTIONAL methods:
//    public void setNIn(InputType inputType, boolean override)
//    public InputPreProcessor getPreProcessorForInputType(InputType inputType)
//    public void applyGlobalConfigToLayer(NeuralNetConfiguration.Builder globalConfig)
}
