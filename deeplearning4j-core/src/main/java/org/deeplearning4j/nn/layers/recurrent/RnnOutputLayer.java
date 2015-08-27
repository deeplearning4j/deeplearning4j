package org.deeplearning4j.nn.layers.recurrent;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BaseOutputLayer;
import org.nd4j.linalg.api.ndarray.INDArray;

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
		if(shape[0]==1) return in.tensorAlongDimension(0,1,2);
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

    /**
     * Returns the f1 score for the given examples.
     * Think of this to be like a percentage right.
     * The higher the number the more it got right.
     * This is on a scale from 0 to 1.
     *
     * @param examples the examples to classify (one example in each row, may be time series)
     * @param labels the true labels (may be time series)
     * @return the scores for each ndarray
     */
    @Override
    public double f1Score(INDArray examples, INDArray labels) {
        Evaluation eval = new Evaluation();
        if(examples.rank() == 3) examples = reshape3dTo2d(examples);
        if(labels.rank() == 3) labels = reshape3dTo2d(labels);
        eval.eval(labels,labelProbabilities(examples));
        return  eval.f1();

    }


//    public  void setLabels(INDArray labels) {
//    	//Reshape labels from 3d to 2d. Similar to RnnToFeedForwardPreprocessor.preProcess()
//    	if(labels != null && labels.rank() == 3) super.setLabels(reshape3dTo2d(labels));
//    	else super.setLabels(labels);
//    }
    
    public INDArray getInput() {
        return input;
    }

    @Override
    public void setInput(INDArray input,boolean training) {
    	if( input != null && input.rank() == 3 ) super.setInput(reshape3dTo2d(input), training);
    	else super.setInput(input, training);
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
