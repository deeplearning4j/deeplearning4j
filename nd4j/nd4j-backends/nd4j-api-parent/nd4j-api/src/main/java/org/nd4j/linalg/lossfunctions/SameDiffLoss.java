/*******************************************************************************
 * Copyright (c) 2020 Konduit K.K.
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
package org.nd4j.linalg.lossfunctions;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;
import java.util.HashMap;
import java.util.Map;


public abstract class SameDiffLoss implements ILossFunction {

    protected SameDiff sd = SameDiff.create();


    protected SameDiffLoss() {


        SDVariable layerInput = sd.placeHolder("layerInput", DataType.FLOAT, -1);
        SDVariable labels = sd.placeHolder("labels", DataType.FLOAT, -1);
        this.defineLoss(sd, layerInput, labels);


    }


    public abstract SDVariable defineLoss(SameDiff sd, SDVariable layerInput, SDVariable labels);

    /**
     * Compute the score (loss function value) for the given inputs.
     *
     * @param labels       Label/expected preOutput
     * @param preOutput    Output of the model (neural network)
     * @param activationFn Activation function that should be applied to preOutput
     * @param mask         Mask array; may be null
     * @param average      Whether the score should be averaged (divided by number of rows in labels/preOutput) or not   @return Loss function value
     */
    public double computeScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, boolean average) {

        // The score overall consists of the
        // sum of the negative log likelihoods for each
        // of the individual labels.


        INDArray scoreArr = computeScoreArray(labels, preOutput, activationFn, mask);

        double score = scoreArr.sumNumber().doubleValue();
        if (average) {
            score /= scoreArr.size(0);
        }
        return score;
    }


    /**
     * Compute the score (loss function value) for each example individually.
     * For input [numExamples,nOut] returns scores as a column vector: [numExamples,1]
     *
     * @param labels       Labels/expected output
     * @param preOutput    Output of the model (neural network)
     * @param activationFn Activation function that should be applied to preOutput
     * @param mask         @return Loss function value for each example; column vector
     */
    public INDArray computeScoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {

        Preconditions.checkArgument((labels.size(1) != preOutput.size(1)), "Labels array numColumns (size(1) = %s) does not match output layer number of outputs (nOut = %s)", labels.size(1), preOutput.size(1));

        INDArray output = activationFn.getActivation(preOutput.dup(), true);

        Map<String, INDArray> m = new HashMap<>();
        m.put("labels", labels);
        m.put("layerInput", output);

        INDArray scoreArr = sd.outputSingle(m,"scoreArr");

        if (mask != null) {
            LossUtil.applyMask(scoreArr, mask);
        }
        return scoreArr;


    }


    /**
     * Compute the gradient of the loss function with respect to the inputs: dL/dOutput
     *
     * @param labels       Label/expected output
     * @param preOutput    Output of the model (neural network), before the activation function is applied
     * @param activationFn Activation function that should be applied to preOutput
     * @param mask         Mask array; may be null
     * @return Gradient dL/dPreOut
     */
    public INDArray computeGradient(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {


        Map<String, INDArray> m = new HashMap<>();
        INDArray output = activationFn.getActivation(preOutput.dup(), true);
        m.put("labels", labels);
        m.put("layerInput", output);

        Map<String, INDArray> grads = sd.calculateGradients(m, "layerInput");

        if (mask != null) {
            LossUtil.applyMask(grads.get("layerInput"), mask);
        }
        return grads.get("layerInput");
    }

    /**
     * Compute both the score (loss function value) and gradient. This is equivalent to calling {@link #computeScore(INDArray, INDArray, IActivation, INDArray, boolean)}
     * and {@link #computeGradient(INDArray, INDArray, IActivation, INDArray)} individually
     *
     * @param labels       Label/expected output
     * @param preOutput    Output of the model (neural network)
     * @param activationFn Activation function that should be applied to preOutput
     * @param mask         Mask array; may be null
     * @param average      Whether the score should be averaged (divided by number of rows in labels/output) or not
     * @return The score (loss function value) and gradient
     */
    public Pair<Double, INDArray> computeGradientAndScore(INDArray labels, INDArray preOutput, IActivation activationFn,
                                                          INDArray mask, boolean average) {

        Pair<Double, INDArray> GradientAndScore = new Pair<>();
        GradientAndScore.setFirst(this.computeScore(labels, preOutput, activationFn, mask, average));
        GradientAndScore.setSecond(this.computeGradient(labels, preOutput, activationFn, mask));

        return GradientAndScore;
    }

    public String name() {
        return getClass().getSimpleName();
    }

}




