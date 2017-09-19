package org.deeplearning4j.nn.conf.preprocessor;

import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.api.activations.Activations;
import org.deeplearning4j.nn.api.activations.ActivationsFactory;
import org.deeplearning4j.nn.api.gradients.Gradients;
import org.deeplearning4j.nn.api.gradients.GradientsFactory;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.util.TimeSeriesUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.primitives.Pair;

import java.util.Arrays;

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
@Slf4j
public class RnnToFeedForwardPreProcessor implements InputPreProcessor {

    @Override
    public Activations preProcess(Activations activations, int miniBatchSize, boolean training) {
        if(activations.size() != 1){
            throw new IllegalArgumentException("Cannot preprocess input: Activations must have exactly 1 array. Got: "
                    + activations.size());
        }
        INDArray input = activations.get(0);
        //Need to reshape RNN activations (3d) activations to 2d (for input into feed forward layer)
        if (input.rank() != 3)
            throw new IllegalArgumentException(
                            "Invalid input: expect NDArray with rank 3 (i.e., activations for RNN layer)");

        if (input.ordering() != 'f')
            input = input.dup('f');

        int[] shape = input.shape();
        INDArray ret = null;
        if (shape[0] == 1) {
            ret = input.tensorAlongDimension(0, 1, 2).permutei(1, 0); //Edge case: miniBatchSize==1
        }
        if (shape[2] == 1) {
            ret = input.tensorAlongDimension(0, 1, 0); //Edge case: timeSeriesLength=1
        }

        if(ret == null){
            INDArray permuted = input.permute(0, 2, 1); //Permute, so we get correct order after reshaping
            ret = permuted.reshape('f', shape[0] * shape[2], shape[1]);
        }

        Pair<INDArray, MaskState> p = feedForwardMaskArray(activations.getMask(0), activations.getMaskState(0), miniBatchSize);
        return ActivationsFactory.getInstance().create(ret, p.getFirst(), p.getSecond());
    }

    @Override
    public Gradients backprop(Gradients gradients, int miniBatchSize) {
        if(gradients.size() != 1){
            throw new IllegalArgumentException("Cannot preprocess activation gradients: Activation gradients must have " +
                    "exactly 1 array. Got: " + gradients.size());
        }

        INDArray output = gradients.get(0);
        if (output == null)
            return null; //In a few cases: output may be null, and this is valid. Like time series data -> embedding layer
        //Need to reshape FeedForward layer epsilons (2d) to 3d (for use in RNN layer backprop calculations)
        if (output.rank() != 2)
            throw new IllegalArgumentException(
                            "Invalid input: expect NDArray with rank 2 (i.e., epsilons from feed forward layer)");
        if (output.ordering() == 'c')
            output = Shape.toOffsetZeroCopy(output, 'f');

        int[] shape = output.shape();
        INDArray reshaped = output.reshape('f', miniBatchSize, shape[0] / miniBatchSize, shape[1]);
        return GradientsFactory.getInstance().create(reshaped.permute(0, 2, 1), gradients.getParameterGradients());
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

    @Override
    public InputType getOutputType(InputType inputType) {
        if (inputType == null || inputType.getType() != InputType.Type.RNN) {
            throw new IllegalStateException("Invalid input: expected input of type RNN, got " + inputType);
        }

        InputType.InputTypeRecurrent rnn = (InputType.InputTypeRecurrent) inputType;
        return InputType.feedForward(rnn.getSize());
    }


    private Pair<INDArray, MaskState> feedForwardMaskArray(INDArray maskArray, MaskState currentMaskState,
                    int minibatchSize) {
        //Assume mask array is 2d for time series (1 value per time step)
        if (maskArray == null) {
            return new Pair<>(maskArray, currentMaskState);
        } else if (maskArray.rank() == 2) {
            //Need to reshape mask array from [minibatch,timeSeriesLength] to [minibatch*timeSeriesLength, 1]
            return new Pair<>(TimeSeriesUtils.reshapeTimeSeriesMaskToVector(maskArray), currentMaskState);
        } else {
            throw new IllegalArgumentException("Received mask array of rank " + maskArray.rank()
                            + "; expected rank 2 mask array. Mask array shape: " + Arrays.toString(maskArray.shape()));
        }
    }
}
