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
import org.nd4j.linalg.api.ndarray.INDArray;

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
		if(shape[0]==1) return in.tensorAlongDimension(0,1,2);	//Edge case: miniBatchSize==1
		if(shape[2]==1) return in.tensorAlongDimension(0,1,0);	//Edge case: timeSeriesLength=1
		INDArray permuted = in.permute(0,2,1);	//Permute, so we get correct order after reshaping
		return permuted.reshape(shape[0]*shape[2],shape[1]);
	}
	
	private INDArray reshape2dTo3d(INDArray in){
		if( in.rank() != 2 ) throw new IllegalArgumentException("Invalid input: expect NDArray with rank 2");
		//Based on: RnnToFeedForwardPreProcessor
		int[] shape = in.shape();
		int miniBatchSize = getInputMiniBatchSize();
		INDArray reshaped = in.reshape(miniBatchSize,shape[0]/miniBatchSize,shape[1]);
		return reshaped.permute(0,2,1);
	}

    @Override
    public Pair<Gradient,INDArray> backpropGradient(INDArray epsilon) {
    	Pair<Gradient,INDArray> gradAndEpsilonNext = super.backpropGradient(epsilon);
    	INDArray epsilon2d = gradAndEpsilonNext.getSecond();
    	INDArray epsilon3d = reshape2dTo3d(epsilon2d);
		return new Pair<>(gradAndEpsilonNext.getFirst(),epsilon3d);
    }

    /**{@inheritDoc}
     */
    public INDArray output(boolean training) {
        INDArray output2d = super.output(training);
        return reshape2dTo3d(output2d);
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
    public INDArray activate(boolean training) {
    	INDArray activations2d = super.activate(training);
    	return reshape2dTo3d(activations2d);
    }

    @Override
    public Type type() {
        return Type.RECURRENT;
    }
    
    @Override
    public INDArray preOutput(INDArray x, boolean training){
    	return reshape2dTo3d(preOutput2d(x,training));
    }
    
    @Override
    protected INDArray preOutput2d(INDArray input, boolean training){
    	if(input.rank()==3) input = reshape3dTo2d(input);
    	return super.preOutput(input,training);
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
}
