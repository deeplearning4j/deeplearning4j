/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.nn.layers.training;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.exception.DL4JInvalidInputException;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BaseOutputLayer;
import org.deeplearning4j.nn.params.CenterLossParamInitializer;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.LossFunction;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.LossFunctions;


/**
 * Center loss is similar to triplet loss except that it enforces
 * intraclass consistency and doesn't require feed forward of multiple
 * examples. Center loss typically converges faster for training
 * ImageNet-based convolutional networks.
 *
 * "If example x is in class Y, ensure that embedding(x) is close to
 * average(embedding(y)) for all examples y in Y"
 *
 * @author Justin Long (@crockpotveggies)
 */
public class CenterLossOutputLayer extends BaseOutputLayer<org.deeplearning4j.nn.conf.layers.OutputLayer> {

    public CenterLossOutputLayer(NeuralNetConfiguration conf) {
        super(conf);
    }

    public CenterLossOutputLayer(NeuralNetConfiguration conf, INDArray input) {
        super(conf, input);
    }

    /** Compute score after labels and input have been set.
     * @param fullNetworkL1 L1 regularization term for the entire network
     * @param fullNetworkL2 L2 regularization term for the entire network
     * @param training whether score should be calculated at train or test time (this affects things like application of
     *                 dropout, etc)
     * @return score (loss function)
     */
    @Override
    public double computeScore( double fullNetworkL1, double fullNetworkL2, boolean training) {
        if( input == null || labels == null )
            throw new IllegalStateException("Cannot calculate score without input and labels");
        this.fullNetworkL1 = fullNetworkL1;
        this.fullNetworkL2 = fullNetworkL2;
        INDArray preOut = preOutput2d(training);

        // center loss has two components
        // the first enforces inter-class dissimilarity, the second intra-class dissimilarity
        // we use these two functions to compute both
        LossFunctions.LossFunction interClassLoss = LossFunctions.LossFunction.MCXENT;
        LossFunctions.LossFunction intraClassLoss = LossFunctions.LossFunction.L2;

        // TODO

        double interClassScore = lossFunction.computeScore(getLabels2d(), preOut, layerConf().getActivationFn(), maskArray, false);
        double intraClassScore = ;

        double score = interClassScore + intraClassScore;

        score += fullNetworkL1 + fullNetworkL2;
        score /= getInputMiniBatchSize();

        this.score = score;

        return score;
    }

    /**Compute the score for each example individually, after labels and input have been set.
     *
     * @param fullNetworkL1 L1 regularization term for the entire network (or, 0.0 to not include regularization)
     * @param fullNetworkL2 L2 regularization term for the entire network (or, 0.0 to not include regularization)
     * @return A column INDArray of shape [numExamples,1], where entry i is the score of the ith example
     */
    @Override
    public INDArray computeScoreForExamples(double fullNetworkL1, double fullNetworkL2){
        if( input == null || labels == null )
            throw new IllegalStateException("Cannot calculate score without input and labels");
        INDArray preOut = preOutput2d(false);

        ILossFunction lossFunction = layerConf().getLossFn();
        //INDArray scoreArray = lossFunction.computeScoreArray(getLabels2d(),preOut,layerConf().getActivationFunction(),maskArray);
        INDArray scoreArray = lossFunction.computeScoreArray(getLabels2d(),preOut,layerConf().getActivationFn(),maskArray);
        double l1l2 = fullNetworkL1 + fullNetworkL2;
        if(l1l2 != 0.0){
            scoreArray.addi(l1l2);
        }
        return scoreArray;
    }

    @Override
    public void computeGradientAndScore() {
        if(input == null || labels == null)
            return;

        INDArray preOut = preOutput2d(true);
        Pair<Gradient,INDArray> pair = getGradientsAndDelta(preOut);
        this.gradient = pair.getFirst();

        score = computeScore(fullNetworkL1,fullNetworkL2,true);
    }

    @Override
    protected void setScoreWithZ(INDArray z) {
//        setScore(z, null);
        throw new RuntimeException("Not yet implemented");
    }

    @Override
    public Pair<Gradient, Double> gradientAndScore() {
        return new Pair<>(gradient(),score());
    }

    @Override
    public Pair<Gradient,INDArray> backpropGradient(INDArray epsilon) {
        Pair<Gradient,INDArray> pair = getGradientsAndDelta(preOutput2d(true));	//Returns Gradient and delta^(this), not Gradient and epsilon^(this-1)
        INDArray delta = pair.getSecond();

        INDArray epsilonNext = params.get(CenterLossParamInitializer.WEIGHT_KEY).mmul(delta.transpose()).transpose();
        return new Pair<>(pair.getFirst(),epsilonNext);
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
    private Pair<Gradient,INDArray> getGradientsAndDelta(INDArray preOut) {
        ILossFunction lossFunction = layerConf().getLossFn();
        INDArray labels2d = getLabels2d();
        if(labels2d.size(1) != preOut.size(1)){
            throw new DL4JInvalidInputException("Labels array numColumns (size(1) = " + labels2d.size(1) + ") does not match output layer"
                + " number of outputs (nOut = " + preOut.size(1) + ")");
        }
        //INDArray delta = lossFunction.computeGradient(labels2d, preOut, layerConf().getActivationFunction(), maskArray);
        INDArray delta = lossFunction.computeGradient(labels2d, preOut, layerConf().getActivationFn(), maskArray);

        Gradient gradient = new DefaultGradient();

        INDArray weightGradView = gradientViews.get(CenterLossParamInitializer.WEIGHT_KEY);
        INDArray biasGradView = gradientViews.get(CenterLossParamInitializer.BIAS_KEY);

        Nd4j.gemm(input,delta,weightGradView,true,false,1.0,0.0);    //Equivalent to:  weightGradView.assign(input.transpose().mmul(delta));
        biasGradView.assign(delta.sum(0));

        gradient.gradientForVariable().put(CenterLossParamInitializer.WEIGHT_KEY,weightGradView);
        gradient.gradientForVariable().put(CenterLossParamInitializer.BIAS_KEY,biasGradView);

        return new Pair<>(gradient, delta);
    }

}
