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
package org.deeplearning4j.nn.layers.recurrent;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BaseOutputLayer;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.util.Dropout;
import org.deeplearning4j.util.TimeSeriesUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.SoftMax;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;

/**Recurrent Neural Network Output Layer.<br>
 * Handles calculation of gradients etc for various objective functions.<br>
 * Functionally the same as OutputLayer, but handles output and label reshaping
 * automatically.<br>
 * Input and output activations are same as other RNN layers: 3 dimensions with shape
 * [miniBatchSize,nIn,timeSeriesLength] and [miniBatchSize,nOut,timeSeriesLength] respectively.
 * @author Alex Black
 * @see BaseOutputLayer, OutputLayer
 */
public class RnnOutputLayer extends BaseOutputLayer<org.deeplearning4j.nn.conf.layers.RnnOutputLayer> {

	public RnnOutputLayer(NeuralNetConfiguration conf) {
		super(conf);
	}

	public RnnOutputLayer(NeuralNetConfiguration conf, INDArray input) {
        super(conf, input);
    }
	
	private INDArray reshape3dTo2d(INDArray in){
		if( in.rank() != 3 ) throw new IllegalArgumentException("Invalid input: expect NDArray with rank 3");
		int[] shape = in.shape();
		if(shape[0]==1) return in.tensorAlongDimension(0,1,2).permutei(1,0);	//Edge case: miniBatchSize==1
		if(shape[2]==1) return in.tensorAlongDimension(0,1,0);	//Edge case: timeSeriesLength=1
		INDArray permuted = in.permute(0, 2, 1);	//Permute, so we get correct order after reshaping
		return permuted.reshape(shape[0] * shape[2], shape[1]);
	}
	
	private INDArray reshape2dTo3d(INDArray in, int miniBatchSize){
		if( in.rank() != 2 ) throw new IllegalArgumentException("Invalid input: expect NDArray with rank 2");
		//Based on: RnnToFeedForwardPreProcessor
		int[] shape = in.shape();
        if(in.ordering() == 'f') in = Shape.toOffsetZeroCopy(in, 'c');
		INDArray reshaped = in.reshape(miniBatchSize, shape[0] / miniBatchSize, shape[1]);
		return reshaped.permute(0, 2, 1);
	}

    @Override
    public Pair<Gradient,INDArray> backpropGradient(INDArray epsilon) {
        if(input.rank() != 3) throw new UnsupportedOperationException("Input is not rank 3");
        INDArray inputTemp = input;
        this.input = reshape3dTo2d(input);
    	Pair<Gradient,INDArray> gradAndEpsilonNext = super.backpropGradient(epsilon);
        this.input = inputTemp;
    	INDArray epsilon2d = gradAndEpsilonNext.getSecond();
    	INDArray epsilon3d = reshape2dTo3d(epsilon2d, input.size(0));
		return new Pair<>(gradAndEpsilonNext.getFirst(),epsilon3d);
    }

    /**{@inheritDoc}
     */
    @Override
    public double f1Score(INDArray examples, INDArray labels) {
        if(examples.rank() == 3) examples = reshape3dTo2d(examples);
        if(labels.rank() == 3) labels = reshape3dTo2d(labels);
        return super.f1Score(examples, labels);
    }
    
    public INDArray getInput() {
        return input;
    }

    @Override
    public Type type() {
        return Type.RECURRENT;
    }
    
    @Override
    public INDArray preOutput(INDArray x, boolean training){
        setInput(x);
        return reshape2dTo3d(preOutput2d(training),input.size(0));
    }

    @Override
    protected INDArray preOutput2d(boolean training){
        if(input.rank() == 3 ) {
            //Case when called from RnnOutputLayer
            INDArray inputTemp = input;
            input = reshape3dTo2d(input);
            INDArray out = super.preOutput(input, training);
            this.input = inputTemp;
            return out;
        } else {
            //Case when called from BaseOutputLayer
            INDArray out = super.preOutput(input, training);
            return out;
        }
    }
    
    @Override
    protected INDArray output2d(INDArray input){
    	return reshape3dTo2d(output(input));
    }
    
    @Override
    protected INDArray getLabels2d(){
    	if(labels.rank()==3) return reshape3dTo2d(labels);
    	return labels;
    }

    @Override
    public INDArray output(INDArray input) {
        if(input.rank() != 3) throw new IllegalArgumentException("Input must be rank 3 (is: " + input.rank());
        //Returns 3d activations from 3d input
        setInput(input);
        return output(false);
    }

    @Override
    public INDArray output(boolean training){
        //Assume that input is 3d
        if(input.rank() != 3 ) throw new IllegalArgumentException("input must be rank 3");
        INDArray preOutput2d = preOutput2d(training);

        if(conf.getLayer().getActivationFunction().equals("softmax")) {
            INDArray out2d = Nd4j.getExecutioner().execAndReturn(new SoftMax(preOutput2d));
            if(maskArray != null){
                out2d.muliColumnVector(maskArray);
            }
            return reshape2dTo3d(out2d,input.size(0));
        }

        if(training)
            applyDropOutIfNecessary(training);
        INDArray origInput = input;
        this.input = reshape3dTo2d(input);
        INDArray out = super.activate(true);
        this.input = origInput;
        return reshape2dTo3d(out,input.size(0));
    }

    @Override
    public INDArray activate(boolean training) {
        if(input.rank() != 3) throw new UnsupportedOperationException("Input must be rank 3");
        INDArray b = getParam(DefaultParamInitializer.BIAS_KEY);
        INDArray W = getParam(DefaultParamInitializer.WEIGHT_KEY);
        if(conf.isUseDropConnect() && training) {
            W = Dropout.applyDropConnect(this, DefaultParamInitializer.WEIGHT_KEY);
        }

        INDArray input2d = reshape3dTo2d(input);

        INDArray act2d = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf.getLayer().getActivationFunction(),
                input2d.mmul(W).addiRowVector(b)));
        return reshape2dTo3d(act2d, input.size(0));
    }

    @Override
    public void setMaskArray(INDArray maskArray) {
        if(maskArray != null && maskArray.size(1) != 1){
            maskArray = TimeSeriesUtils.reshapeTimeSeriesMaskToVector(maskArray);
        }
        this.maskArray = maskArray;
    }
}
