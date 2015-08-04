package org.deeplearning4j.nn.conf.preprocessor;

import lombok.EqualsAndHashCode;

import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.nd4j.linalg.api.ndarray.INDArray;

/**A preprocessor to allow RNN and feed-forward network layers to be used together.<br>
 * For example, DenseLayer -> GravesLSTM<br>
 * This does two things:<br>
 * (a) Reshapes activations out of FeedFoward layer (which is 2D with shape 
 * [miniBatchSize*timeSeriesLength,layerSize]) into 3d activations (with shape
 * [miniBatchSize,layerSize,timeSeriesLength]) suitable to feed into RNN layers.<br>
 * (b) Reshapes 3d epsilons (weights*deltas from RNN layer, with shape
 * [miniBatchSize,layerSize,timeSeriesLength]) into 2d epsilons (with shape
 * [miniBatchSize*timeSeriesLength,layerSize]) for use in feed forward layer
 * @author Alex Black
 * @see RnnToFeedForwardPreProcessor for opposite case (i.e., GravesLSTM -> DenseLayer etc)
 */
@EqualsAndHashCode
public class FeedForwardToRnnPreProcessor implements InputPreProcessor {
	private static final long serialVersionUID = -9162841658222982319L;
	private final int miniBatchSize;
	private final int rnnLayerSize;
	private final int timeSeriesLength;
	
	/**@param miniBatchSize mini-batch size for training data
	 * @param rnnLayerSize Size (number of hidden units) for the RNN layer
	 * @param timeSeriesLength Length of time series training data
	 */
	public FeedForwardToRnnPreProcessor(int miniBatchSize, int rnnLayerSize, int timeSeriesLength ){
		this.miniBatchSize = miniBatchSize;
		this.rnnLayerSize = rnnLayerSize;
		this.timeSeriesLength = timeSeriesLength;
	}

	@Override
	public INDArray preProcess(INDArray input) {
		//Need to reshape FF activations (2d) activations to 3d (for input into RNN layer)
		if( input.rank() != 2 ) throw new IllegalArgumentException("Invalid input: expect NDArray with rank 2 (i.e., activations for FF layer)");
		
		INDArray reshaped = input.reshape(miniBatchSize,timeSeriesLength,rnnLayerSize);
		return reshaped.permute(0,2,1);
	}

	@Override
	public INDArray backprop(INDArray output) {
		//Need to reshape RNN epsilons (3d) to 2d (for use in FF layer backprop calculations)
		if( output.rank() != 3 ) throw new IllegalArgumentException("Invalid input: expect NDArray with rank 3 (i.e., epsilons from RNN layer)");
		
		INDArray permuted = output.permute(0,2,1);	//Permute, so we get correct order after reshaping
		//TODO: TEMPORARY copy to work around a bug in reshape on permuted NDArrays.
		//This can be removed at some point in the future
		permuted = permuted.dup();
		
		return permuted.reshape(miniBatchSize*timeSeriesLength,rnnLayerSize);
	}

}
