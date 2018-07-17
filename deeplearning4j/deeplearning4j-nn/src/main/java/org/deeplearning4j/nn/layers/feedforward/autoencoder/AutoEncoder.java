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

package org.deeplearning4j.nn.layers.feedforward.autoencoder;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.layers.BasePretrainNetwork;
import org.deeplearning4j.nn.params.PretrainParamInitializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.nn.workspace.ArrayType;

/**
 *  Autoencoder.
 * Add Gaussian noise to input and learn
 * a reconstruction function.
 *
 * @author Adam Gibson
 *
 */
public class AutoEncoder extends BasePretrainNetwork<org.deeplearning4j.nn.conf.layers.AutoEncoder> {

    public AutoEncoder(NeuralNetConfiguration conf) {
        super(conf);
    }

    public AutoEncoder(NeuralNetConfiguration conf, INDArray input) {
        super(conf, input);
    }

    @Override
    public Pair<INDArray, INDArray> sampleHiddenGivenVisible(INDArray v) {
        setInput(v, LayerWorkspaceMgr.noWorkspaces());      //TODO
        INDArray ret = encode(v, true, LayerWorkspaceMgr.noWorkspaces());   //TODO
        return new Pair<>(ret, ret);
    }

    @Override
    public Pair<INDArray, INDArray> sampleVisibleGivenHidden(INDArray h) {
        INDArray ret = decode(h, LayerWorkspaceMgr.noWorkspaces()); //TODO
        return new Pair<>(ret, ret);
    }

    // Encode
    public INDArray encode(INDArray v, boolean training, LayerWorkspaceMgr workspaceMgr) {
        INDArray W = getParamWithNoise(PretrainParamInitializer.WEIGHT_KEY, training, workspaceMgr);
        INDArray hBias = getParamWithNoise(PretrainParamInitializer.BIAS_KEY, training, workspaceMgr);
        INDArray ret = workspaceMgr.createUninitialized(ArrayType.ACTIVATIONS, v.size(0), W.size(1));
        INDArray preAct = v.mmuli(W, ret).addiRowVector(hBias);
        ret = layerConf().getActivationFn().getActivation(preAct, training);
        return workspaceMgr.leverageTo(ArrayType.ACTIVATIONS, ret);
    }

    // Decode
    public INDArray decode(INDArray y, LayerWorkspaceMgr workspaceMgr) {
        INDArray W = getParamWithNoise(PretrainParamInitializer.WEIGHT_KEY, true, workspaceMgr);
        INDArray vBias = getParamWithNoise(PretrainParamInitializer.VISIBLE_BIAS_KEY, true, workspaceMgr);
        INDArray preAct = y.mmul(W.transposei()).addiRowVector(vBias);
        return workspaceMgr.leverageTo(ArrayType.ACTIVATIONS, layerConf().getActivationFn().getActivation(preAct, true));

    }

    @Override
    public INDArray activate(INDArray input, boolean training, LayerWorkspaceMgr workspaceMgr) {
        setInput(input, workspaceMgr);
        return encode(input, training, workspaceMgr);
    }

    @Override
    public boolean isPretrainLayer() {
        return true;
    }

    @Override
    public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr) {
        return encode(input, training, workspaceMgr);
    }

    @Override
    public void computeGradientAndScore(LayerWorkspaceMgr workspaceMgr) {
        INDArray W = getParamWithNoise(PretrainParamInitializer.WEIGHT_KEY, true, workspaceMgr);

        double corruptionLevel = layerConf().getCorruptionLevel();

        INDArray corruptedX = corruptionLevel > 0 ? getCorruptedInput(input, corruptionLevel) : input;
        setInput(corruptedX, workspaceMgr);

        INDArray y = encode(corruptedX, true, workspaceMgr);
        INDArray z = decode(y, workspaceMgr);

        INDArray visibleLoss = input.sub(z);
        INDArray hiddenLoss = layerConf().getSparsity() == 0 ? visibleLoss.mmul(W).muli(y).muli(y.rsub(1))
                        : visibleLoss.mmul(W).muli(y).muli(y.add(-layerConf().getSparsity()));

        INDArray wGradient = corruptedX.transposei().mmul(hiddenLoss).addi(visibleLoss.transposei().mmul(y));
        INDArray hBiasGradient = hiddenLoss.sum(0);
        INDArray vBiasGradient = visibleLoss.sum(0);

        gradient = createGradient(wGradient, vBiasGradient, hBiasGradient);
        setScoreWithZ(z);

    }


}
