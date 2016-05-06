package org.deeplearning4j.nn.conf.preprocessor;

import lombok.Data;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;

/**
 * A preprocessor to allow RNN and feed-forward network layers to be used together.<br>
 * For example, GravesLSTM -> OutputLayer or GravesLSTM -> DenseLayer<br>
 * This does two things:<br>
 * (a) Reshapes activations out of RNN layer (which is 3D with shape
 * [miniBatchSize,layerSize,timeSeriesLength]) into 2d activations (with shape
 * [miniBatchSize*timeSeriesLength,layerSize]) suitable for use in feed-forward layers.<br>
 * (b) Reshapes 2d epsilons (weights*deltas from feed forward layer, with shape
 * [miniBatchSize*timeSeriesLength,layerSize]) into 3d epsilons (with shape
 * [miniBatchSize,layerSize,timeSeriesLength]) for use in RNN layer
 *
 * @author Alex Black
 * @see FeedForwardToRnnPreProcessor for opposite case (i.e., DenseLayer -> GravesLSTM etc)
 */
@Data
public class RnnToFeedForwardPreProcessor implements InputPreProcessor {

    @Override
    public INDArray preProcess(INDArray input, int miniBatchSize) {
        //Need to reshape RNN activations (3d) activations to 2d (for input into feed forward layer)
        if (input.rank() != 3)
            throw new IllegalArgumentException("Invalid input: expect NDArray with rank 3 (i.e., activations for RNN layer)");

        if (input.ordering() != 'f') input = input.dup('f');

        int[] shape = input.shape();
        if (shape[0] == 1) return input.tensorAlongDimension(0, 1, 2).permutei(1, 0);    //Edge case: miniBatchSize==1
        if (shape[2] == 1) return input.tensorAlongDimension(0, 1, 0);    //Edge case: timeSeriesLength=1
        INDArray permuted = input.permute(0, 2, 1);    //Permute, so we get correct order after reshaping
        return permuted.reshape('f', shape[0] * shape[2], shape[1]);
    }

    @Override
    public INDArray backprop(INDArray output, int miniBatchSize) {
        if (output == null)
            return null;    //In a few cases: output may be null, and this is valid. Like time series data -> embedding layer
        //Need to reshape FeedForward layer epsilons (2d) to 3d (for use in RNN layer backprop calculations)
        if (output.rank() != 2)
            throw new IllegalArgumentException("Invalid input: expect NDArray with rank 2 (i.e., epsilons from feed forward layer)");
        if (output.ordering() == 'c') output = Shape.toOffsetZeroCopy(output, 'f');

        int[] shape = output.shape();
        INDArray reshaped = output.reshape('f', miniBatchSize, shape[0] / miniBatchSize, shape[1]);
        return reshaped.permute(0, 2, 1);
    }

    @Override
    public RnnToFeedForwardPreProcessor clone() {
        try {
            RnnToFeedForwardPreProcessor clone = (RnnToFeedForwardPreProcessor) super.clone();
            return clone;
        } catch (CloneNotSupportedException e) {
            throw new RuntimeException(e);
        }
    }
}
