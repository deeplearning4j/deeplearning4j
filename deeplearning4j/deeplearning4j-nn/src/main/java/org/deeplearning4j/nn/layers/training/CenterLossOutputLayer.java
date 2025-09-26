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

package org.deeplearning4j.nn.layers.training;

import org.deeplearning4j.exception.DL4JInvalidInputException;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BaseOutputLayer;
import org.deeplearning4j.nn.params.CenterLossParamInitializer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.common.primitives.Pair;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.nn.workspace.ArrayType;


public class CenterLossOutputLayer extends BaseOutputLayer<org.deeplearning4j.nn.conf.layers.CenterLossOutputLayer> {

    private double fullNetRegTerm;

    public CenterLossOutputLayer(NeuralNetConfiguration conf, DataType dataType) {
        super(conf, dataType);
    }

    /** Compute score after labels and input have been set.
     * @param fullNetRegTerm Regularization score term for the entire network
     * @param training whether score should be calculated at train or test time (this affects things like application of
     *                 dropout, etc)
     * @return score (loss function)
     */
    @Override
    public double computeScore(double fullNetRegTerm, boolean training, LayerWorkspaceMgr workspaceMgr) {
        if (input == null || labels == null)
            throw new IllegalStateException("Cannot calculate score without input and labels " + layerId());
        this.fullNetRegTerm = fullNetRegTerm;
        INDArray preOut = preOutput2d(training, workspaceMgr);

        // center loss has two components
        // the first enforces inter-class dissimilarity, the second intra-class dissimilarity (squared l2 norm of differences)
        ILossFunction interClassLoss = layerConf().getLossFn();

        // calculate the intra-class score component
        INDArray centers = params.get(CenterLossParamInitializer.CENTER_KEY);
        INDArray l = labels.castTo(centers.dataType()); //Ensure correct dtype (same as params); no-op if already correct dtype
        INDArray centersForExamples = l.mmul(centers);

        //        double intraClassScore = intraClassLoss.computeScore(centersForExamples, input, Activation.IDENTITY.getActivationFunction(), maskArray, false);
        INDArray norm2DifferenceSquared = input.sub(centersForExamples).norm2(1);
        norm2DifferenceSquared.muli(norm2DifferenceSquared);

        double sum = norm2DifferenceSquared.sumNumber().doubleValue();
        double lambda = layerConf().getLambda();
        double intraClassScore = lambda / 2.0 * sum;

        //        intraClassScore = intraClassScore * layerConf().getLambda() / 2;

        // now calculate the inter-class score component
        double interClassScore = interClassLoss.computeScore(getLabels2d(workspaceMgr, ArrayType.FF_WORKING_MEM), preOut, layerConf().getActivationFn(),
                        maskArray, false);

        double score = interClassScore + intraClassScore;

        score /= getInputMiniBatchSize();
        score += fullNetRegTerm;

        this.score = score;
        return score;
    }

    /**Compute the score for each example individually, after labels and input have been set.
     *
     * @param fullNetRegTerm Regularization term for the entire network (or, 0.0 to not include regularization)
     * @return A column INDArray of shape [numExamples,1], where entry i is the score of the ith example
     */
    @Override
    public INDArray computeScoreForExamples(double fullNetRegTerm, LayerWorkspaceMgr workspaceMgr) {
        if (input == null || labels == null)
            throw new IllegalStateException("Cannot calculate score without input and labels " + layerId());
        INDArray preOut = preOutput2d(false, workspaceMgr);

        // calculate the intra-class score component
        INDArray centers = params.get(CenterLossParamInitializer.CENTER_KEY);
        INDArray centersForExamples = labels.mmul(centers);
        INDArray intraClassScoreArray = input.sub(centersForExamples);

        // calculate the inter-class score component
        ILossFunction interClassLoss = layerConf().getLossFn();
        INDArray scoreArray = interClassLoss.computeScoreArray(getLabels2d(workspaceMgr, ArrayType.FF_WORKING_MEM), preOut, layerConf().getActivationFn(),
                        maskArray);
        scoreArray.addi(intraClassScoreArray.muli(layerConf().getLambda() / 2));

        if (fullNetRegTerm != 0.0) {
            scoreArray.addi(fullNetRegTerm);
        }
        return workspaceMgr.leverageTo(ArrayType.ACTIVATIONS, scoreArray);
    }

    @Override
    public void computeGradientAndScore(LayerWorkspaceMgr workspaceMgr) {
        if (input == null || labels == null)
            return;

        INDArray preOut = preOutput2d(true, workspaceMgr);
        Pair<Gradient, INDArray> pair = getGradientsAndDelta(preOut, workspaceMgr);
        this.gradient = pair.getFirst();

        score = computeScore(fullNetRegTerm, true, workspaceMgr);
    }

    @Override
    protected void setScoreWithZ(INDArray z) {
        throw new RuntimeException("Not supported " + layerId());
    }

    @Override
    public Pair<Gradient, Double> gradientAndScore() {
        return new Pair<>(gradient(), score());
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(true);
        Pair<Gradient, INDArray> pair = getGradientsAndDelta(preOutput2d(true, workspaceMgr), workspaceMgr); //Returns Gradient and delta^(this), not Gradient and epsilon^(this-1)
        INDArray delta = pair.getSecond();

        // centers
        INDArray centers = params.get(CenterLossParamInitializer.CENTER_KEY);
        INDArray l = labels.castTo(centers.dataType());     //Ensure correct dtype (same as params); no-op if already correct dtype
        INDArray centersForExamples = l.mmul(centers);
        INDArray dLcdai = input.sub(centersForExamples);

        INDArray w = getParamWithNoise(CenterLossParamInitializer.WEIGHT_KEY, true, workspaceMgr);

        INDArray epsilonNext = workspaceMgr.createUninitialized(ArrayType.ACTIVATION_GRAD, w.dataType(), new long[]{w.size(0), delta.size(0)}, 'f');
        epsilonNext = w.mmuli(delta.transpose(), epsilonNext).transpose();
        double lambda = layerConf().getLambda();
        epsilonNext.addi(dLcdai.muli(lambda)); // add center loss here

        weightNoiseParams.clear();

        return new Pair<>(pair.getFirst(), epsilonNext);
    }

    /**
     * Gets the gradient from one training iteration
     * @return the gradient (bias and weight matrix)
     */
    @Override
    public Gradient gradient() {
        return gradient;
    }

    /** Returns tuple: {Gradient,Delta,Output} given preOut */
    private Pair<Gradient, INDArray> getGradientsAndDelta(INDArray preOut, LayerWorkspaceMgr workspaceMgr) {
        ILossFunction lossFunction = layerConf().getLossFn();
        INDArray labels2d = getLabels2d(workspaceMgr, ArrayType.BP_WORKING_MEM);
        if (labels2d.size(1) != preOut.size(1)) {
            throw new DL4JInvalidInputException(
                            "Labels array numColumns (size(1) = " + labels2d.size(1) + ") does not match output layer"
                                            + " number of outputs (nOut = " + preOut.size(1) + ") " + layerId());
        }

        INDArray delta = lossFunction.computeGradient(labels2d, preOut, layerConf().getActivationFn(), maskArray);

        Gradient gradient = new DefaultGradient();

        INDArray weightGradView = gradientViews.get(CenterLossParamInitializer.WEIGHT_KEY);
        INDArray biasGradView = gradientViews.get(CenterLossParamInitializer.BIAS_KEY);
        INDArray centersGradView = gradientViews.get(CenterLossParamInitializer.CENTER_KEY);

        // centers delta
        double alpha = layerConf().getAlpha();

        INDArray centers = params.get(CenterLossParamInitializer.CENTER_KEY);
        INDArray l = labels.castTo(centers.dataType()); //Ensure correct dtype (same as params); no-op if already correct dtype
        INDArray centersForExamples = l.mmul(centers);
        INDArray diff = centersForExamples.sub(input).muli(alpha);
        INDArray numerator = l.transpose().mmul(diff);
        INDArray denominator = l.sum(0).reshape(l.size(1), 1).addi(1.0);

        INDArray deltaC;
        if (layerConf().getGradientCheck()) {
            double lambda = layerConf().getLambda();
            //For gradient checks: need to multiply dLc/dcj by lambda to get dL/dcj
            deltaC = numerator.muli(lambda);
        } else {
            deltaC = numerator.diviColumnVector(denominator);
        }
        centersGradView.assign(deltaC);



        // other standard calculations
        Nd4j.gemm(input, delta, weightGradView, true, false, 1.0, 0.0); //Equivalent to:  weightGradView.assign(input.transpose().mmul(delta));
        delta.sum(biasGradView, 0); //biasGradView is initialized/zeroed first in sum op

        gradient.gradientForVariable().put(CenterLossParamInitializer.WEIGHT_KEY, weightGradView);
        gradient.gradientForVariable().put(CenterLossParamInitializer.BIAS_KEY, biasGradView);
        gradient.gradientForVariable().put(CenterLossParamInitializer.CENTER_KEY, centersGradView);

        return new Pair<>(gradient, delta);
    }

    @Override
    protected INDArray getLabels2d(LayerWorkspaceMgr workspaceMgr, ArrayType arrayType) {
        return labels;
    }
}
