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

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.samediff.SDLayerParams;
import org.deeplearning4j.nn.conf.layers.samediff.SameDiffOutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.weights.WeightInitUtil;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Map;

public class SameDiffMSEOutputLayer extends SameDiffOutputLayer {

    private int nIn;
    private int nOut;
    private Activation activation;
    private WeightInit weightInit;

    public SameDiffMSEOutputLayer(int nIn, int nOut, Activation activation, WeightInit weightInit){
        this.nIn = nIn;
        this.nOut = nOut;
        this.activation = activation;
        this.weightInit = weightInit;
    }

    @Override
    public SDVariable defineLayer(SameDiff sameDiff, SDVariable layerInput, SDVariable labels, Map<String, SDVariable> paramTable) {
        SDVariable z = sameDiff.mmul(layerInput, paramTable.get("W")).add(paramTable.get("b"));
        SDVariable out = activation.asSameDiff("out", sameDiff, z);
        //MSE: 1/nOut * (input-labels)^2
        SDVariable diff = out.sub(labels);
        return diff.mul(diff).mean(1).sum();
    }

    @Override
    public String activationsVertexName() {
        return "out";
    }

    @Override
    public void defineParameters(SDLayerParams params) {
        params.addWeightParam("W", nIn, nOut);
        params.addBiasParam("b", 1, nOut);
    }

    @Override
    public void initializeParameters(Map<String, INDArray> params) {
        WeightInitUtil.initWeights(nIn, nOut, new long[]{nIn, nOut}, weightInit, null, 'f', params.get("W"));
        params.get("b").assign(0.0);
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        return InputType.feedForward(nOut);
    }

    @Override
    public char paramReshapeOrder(String param){
        //To match DL4J for easy comparison
        return 'f';
    }

    @Override
    public void applyGlobalConfigToLayer(NeuralNetConfiguration.Builder globalConfig){

    }

}
