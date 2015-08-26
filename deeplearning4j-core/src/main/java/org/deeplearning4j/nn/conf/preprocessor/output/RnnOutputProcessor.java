package org.deeplearning4j.nn.conf.preprocessor.output;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;

/** RnnOutputProcessor does two things:<br>
 * (a) reshapes 2d output from OutputLayer to 3d (time series) output,
 * in the same manner FeedForwardToRnnPreProcessor<br>
 * (b) reshapes 3d (time series) labels for use with OutputLayer
 * (similar to what FeedForwardToRnnPreprocessor.backprop() does)
 * @author Alex Black
 *
 */
public class RnnOutputProcessor implements NetworkOutputProcessor {

	@Override
	public INDArray processOutput(INDArray output, MultiLayerNetwork network) {
		//Reshape output from 2d to 3d, as per FeedForwardToRnnPreprocessor.preProcess()
		int[] shape = output.shape();
		int miniBatchSize = network.getInputMiniBatchSize();
		INDArray reshaped = output.reshape(miniBatchSize,shape[0]/miniBatchSize,shape[1]);
		return reshaped.permute(0,2,1);
	}

	@Override
	public INDArray processLabels(INDArray labels, MultiLayerNetwork network) {
		//Reshape from 3d to 2d, as per FeedForwardToRnnPreprocessor.backprop()
		int[] shape = labels.shape();
		if(shape[0]==1) return labels.tensorAlongDimension(0,1,2); //Edge case: miniBatchsize=1
		INDArray permuted = labels.permute(0,2,1);	//Permute, so we get correct order after reshaping
		return permuted.reshape(shape[0]*shape[2],shape[1]);
	}
}
