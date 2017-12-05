/*-
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
package org.deeplearning4j.nn.layers.recurrent;

import lombok.Setter;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.api.layers.IOutputLayer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.deeplearning4j.nn.layers.BaseOutputLayer;
import org.deeplearning4j.nn.layers.LossLayer;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.util.TimeSeriesUtils;
import org.nd4j.linalg.activations.impl.ActivationSoftmax;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.SoftMax;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.primitives.Pair;

import java.util.Arrays;
import java.util.List;

/**Recurrent Neural Network Output Layer.<br>
 * Handles calculation of gradients etc for various objective functions.<br>
 * Functionally the same as OutputLayer, but handles output and label reshaping
 * automatically.<br>
 * Input and output activations are same as other RNN layers: 3 dimensions with shape
 * [miniBatchSize,nIn,timeSeriesLength] and [miniBatchSize,nOut,timeSeriesLength] respectively.
 * @author Alex Black
 * @see RnnOutputLayer
 */
public class RnnLossLayer extends BaseLayer<org.deeplearning4j.nn.conf.layers.RnnLossLayer> implements IOutputLayer {

    @Setter protected INDArray labels;
    private double fullNetworkL1;
    private double fullNetworkL2;

    public RnnLossLayer(NeuralNetConfiguration conf) {
        super(conf);
    }

    public RnnLossLayer(NeuralNetConfiguration conf, INDArray input) {
        super(conf, input);
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon) {
        if (input.rank() != 3)
            throw new UnsupportedOperationException(
                            "Input is not rank 3. Got input with rank " + input.rank() + " " + layerId());
        INDArray inputTemp = input;
        this.input = TimeSeriesUtils.reshape3dTo2d(input);
        Pair<Gradient, INDArray> gradAndEpsilonNext = super.backpropGradient(epsilon);
        this.input = inputTemp;
        INDArray epsilon2d = gradAndEpsilonNext.getSecond();
        INDArray epsilon3d = TimeSeriesUtils.reshape2dTo3d(epsilon2d, input.size(0));

        weightNoiseParams.clear();

        return new Pair<>(gradAndEpsilonNext.getFirst(), epsilon3d);
    }

    @Override
    public double f1Score(DataSet data) {
        return 0;
    }

    /**{@inheritDoc}
     */
    @Override
    public double f1Score(INDArray examples, INDArray labels) {
        if (examples.rank() == 3)
            examples = TimeSeriesUtils.reshape3dTo2d(examples);
        if (labels.rank() == 3)
            labels = TimeSeriesUtils.reshape3dTo2d(labels);
        return f1Score(examples, labels);
    }

    @Override
    public int numLabels() {
        return 0;
    }

    @Override
    public void fit(DataSetIterator iter) {

    }

    @Override
    public int[] predict(INDArray examples) {
        return new int[0];
    }

    @Override
    public List<String> predict(DataSet dataSet) {
        return null;
    }

    @Override
    public INDArray labelProbabilities(INDArray examples) {
        return null;
    }

    @Override
    public void fit(INDArray examples, INDArray labels) {

    }

    @Override
    public void fit(DataSet data) {

    }

    @Override
    public void fit(INDArray examples, int[] labels) {

    }

    public INDArray getInput() {
        return input;
    }

    @Override
    public Type type() {
        return Type.RECURRENT;
    }

    @Override
    public INDArray preOutput(INDArray x, boolean training) {
        setInput(x);
        return TimeSeriesUtils.reshape2dTo3d(preOutput2d(training), input.size(0));
    }

    protected INDArray preOutput2d(boolean training) {
        if (input.rank() == 3) {
            //Case when called from RnnOutputLayer
            INDArray inputTemp = input;
            input = TimeSeriesUtils.reshape3dTo2d(input);
            INDArray out = super.preOutput(input, training);
            this.input = inputTemp;
            return out;
        } else {
            //Case when called from BaseOutputLayer
            INDArray out = super.preOutput(input, training);
            return out;
        }
    }


    protected INDArray getLabels2d() {
        if (labels.rank() == 3)
            return TimeSeriesUtils.reshape3dTo2d(labels);
        return labels;
    }


    public INDArray output(INDArray input) {
        if (input.rank() != 3)
            throw new IllegalArgumentException("Input must be rank 3 (is: " + input.rank() + ") " + layerId());
        //Returns 3d activations from 3d input
        setInput(input);
        return output(false);
    }


    public INDArray output(boolean training) {
        //Assume that input is 3d
        if (input.rank() != 3)
            throw new IllegalArgumentException(
                            "input must be rank 3. Got input with rank " + input.rank() + " " + layerId());
        INDArray preOutput2d = preOutput2d(training);

        //if(conf.getLayer().getActivationFunction().equals("softmax")) {
        if (layerConf().getActivationFn() instanceof ActivationSoftmax) {
            INDArray out2d = Nd4j.getExecutioner().execAndReturn(new SoftMax(preOutput2d));
            if (maskArray != null) {
                out2d.muliColumnVector(maskArray);
            }
            return TimeSeriesUtils.reshape2dTo3d(out2d, input.size(0));
        }

        applyDropOutIfNecessary(training);
        INDArray origInput = input;
        this.input = TimeSeriesUtils.reshape3dTo2d(input);
        INDArray out = super.activate(true);
        this.input = origInput;
        if (maskArray != null) {
            out.muliColumnVector(maskArray);
        }
        return TimeSeriesUtils.reshape2dTo3d(out, input.size(0));
    }

    @Override
    public INDArray activate(boolean training) {
        if (input.rank() != 3)
            throw new UnsupportedOperationException(
                            "Input must be rank 3. Got input with rank " + input.rank() + " " + layerId());
        INDArray b = getParamWithNoise(DefaultParamInitializer.BIAS_KEY, training);
        INDArray W = getParamWithNoise(DefaultParamInitializer.WEIGHT_KEY, training);

        INDArray input2d = TimeSeriesUtils.reshape3dTo2d(input);

        //INDArray act2d = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf.getLayer().getActivationFunction(),
        //        input2d.mmul(W).addiRowVector(b)));
        INDArray act2d = layerConf().getActivationFn().getActivation(input2d.mmul(W).addiRowVector(b), training);
        if (maskArray != null) {
            if(!maskArray.isColumnVector() || Arrays.equals(maskArray.shape(), act2d.shape())){
                //Per output masking
                act2d.muli(maskArray);
            } else {
                //Per time step masking
                act2d.muliColumnVector(maskArray);
            }
        }
        return TimeSeriesUtils.reshape2dTo3d(act2d, input.size(0));
    }

    @Override
    public void setMaskArray(INDArray maskArray) {
        if (maskArray != null) {
            //Two possible cases:
            //(a) per time step masking - rank 2 mask array -> reshape to rank 1 (column vector)
            //(b) per output masking - rank 3 mask array  -> reshape to rank 2 (
            if (maskArray.rank() == 2) {
                this.maskArray = TimeSeriesUtils.reshapeTimeSeriesMaskToVector(maskArray);
            } else if (maskArray.rank() == 3) {
                this.maskArray = TimeSeriesUtils.reshape3dTo2d(maskArray);
            } else {
                throw new UnsupportedOperationException(
                                "Invalid mask array: must be rank 2 or 3 (got: rank " + maskArray.rank() + ", shape = "
                                                + Arrays.toString(maskArray.shape()) + ") " + layerId());
            }
        } else {
            this.maskArray = null;
        }
    }

    @Override
    public boolean isPretrainLayer() {
        return false;
    }

    @Override
    public Pair<INDArray, MaskState> feedForwardMaskArray(INDArray maskArray, MaskState currentMaskState,
                    int minibatchSize) {
        this.maskArray = TimeSeriesUtils.reshapeTimeSeriesMaskToVector(maskArray);
        this.maskState = currentMaskState;

        return null; //Last layer in network
    }

    @Override
    public void setLabels(INDArray labels) {

    }

    @Override
    public INDArray getLabels() {
        return null;
    }

    @Override
    public double computeScore(double fullNetworkL1, double fullNetworkL2, boolean training) {
        return 0;
    }

    /**Compute the score for each example individually, after labels and input have been set.
     *
     * @param fullNetworkL1 L1 regularization term for the entire network (or, 0.0 to not include regularization)
     * @param fullNetworkL2 L2 regularization term for the entire network (or, 0.0 to not include regularization)
     * @return A column INDArray of shape [numExamples,1], where entry i is the score of the ith example
     */
    @Override
    public INDArray computeScoreForExamples(double fullNetworkL1, double fullNetworkL2) {
        //For RNN: need to sum up the score over each time step before returning.

        if (input == null || labels == null)
            throw new IllegalStateException("Cannot calculate score without input and labels " + layerId());
        INDArray preOut = preOutput2d(false);

        ILossFunction lossFunction = layerConf().getLossFn();
        INDArray scoreArray =
                        lossFunction.computeScoreArray(getLabels2d(), preOut, layerConf().getActivationFn(), maskArray);
        //scoreArray: shape [minibatch*timeSeriesLength, 1]
        //Reshape it to [minibatch, timeSeriesLength] then sum over time step

        INDArray scoreArrayTs = TimeSeriesUtils.reshapeVectorToTimeSeriesMask(scoreArray, input.size(0));
        INDArray summedScores = scoreArrayTs.sum(1);

        double l1l2 = fullNetworkL1 + fullNetworkL2;
        if (l1l2 != 0.0) {
            summedScores.addi(l1l2);
        }

        return summedScores;
    }
}
