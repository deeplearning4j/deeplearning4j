package org.deeplearning4j.nn.conf.preprocessor;

import lombok.*;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.api.activations.Activations;
import org.deeplearning4j.nn.api.activations.ActivationsFactory;
import org.deeplearning4j.nn.api.gradients.Gradients;
import org.deeplearning4j.nn.api.gradients.GradientsFactory;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.util.TimeSeriesUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.shade.jackson.annotation.JsonCreator;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.Arrays;

/**A preprocessor to allow CNN and RNN layers to be used together.<br>
 * For example, ConvolutionLayer -> GravesLSTM
 * Functionally equivalent to combining CnnToFeedForwardPreProcessor + FeedForwardToRnnPreProcessor<br>
 * Specifically, this does two things:<br>
 * (a) Reshape 4d activations out of CNN layer, with shape [timeSeriesLength*miniBatchSize, numChannels, inputHeight, inputWidth])
 * into 3d (time series) activations (with shape [numExamples, inputHeight*inputWidth*numChannels, timeSeriesLength])
 * for use in RNN layers<br>
 * (b) Reshapes 3d epsilons (weights.*deltas) out of RNN layer (with shape
 * [miniBatchSize,inputHeight*inputWidth*numChannels,timeSeriesLength]) into 4d epsilons with shape
 * [miniBatchSize*timeSeriesLength, numChannels, inputHeight, inputWidth] suitable to feed into CNN layers.
 * Note: numChannels is equivalent to depth or featureMaps referenced in different literature
 * @author Alex Black
 */
@Data
@EqualsAndHashCode(exclude = {"product"})
public class CnnToRnnPreProcessor implements InputPreProcessor {
    private int inputHeight;
    private int inputWidth;
    private int numChannels;

    @Getter(AccessLevel.NONE)
    @Setter(AccessLevel.NONE)
    private int product;

    @JsonCreator
    public CnnToRnnPreProcessor(@JsonProperty("inputHeight") int inputHeight,
                    @JsonProperty("inputWidth") int inputWidth, @JsonProperty("numChannels") int numChannels) {
        this.inputHeight = inputHeight;
        this.inputWidth = inputWidth;
        this.numChannels = numChannels;
        this.product = inputHeight * inputWidth * numChannels;
    }

    @Override
    public Activations preProcess(Activations a, int miniBatchSize, boolean training) {
        if(a.size() != 1){
            throw new IllegalArgumentException("Cannot preprocess input: Activations must have exactly 1 array. Got: "
                    + a.size());
        }
        INDArray input = a.get(0);
        if (input.rank() != 4)
            throw new IllegalArgumentException(
                            "Invalid input: expect CNN activations with rank 4 (received input with shape "
                                            + Arrays.toString(input.shape()) + ")");
        //Input: 4d activations (CNN)
        //Output: 3d activations (RNN)

        if (input.ordering() != 'c')
            input = input.dup('c');

        int[] shape = input.shape(); //[timeSeriesLength*miniBatchSize, numChannels, inputHeight, inputWidth]

        //First: reshape 4d to 2d, as per CnnToFeedForwardPreProcessor
        INDArray twod = input.reshape('c', input.size(0), ArrayUtil.prod(input.shape()) / input.size(0));
        //Second: reshape 2d to 3d, as per FeedForwardToRnnPreProcessor
        INDArray reshaped = twod.dup('f').reshape('f', miniBatchSize, shape[0] / miniBatchSize, product);
        INDArray ret = reshaped.permute(0, 2, 1);

        Pair<INDArray, MaskState> p = feedForwardMaskArray(a.getMask(0), a.getMaskState(0), miniBatchSize);
        return ActivationsFactory.getInstance().create(ret, p.getFirst(), p.getSecond());
    }

    @Override
    public Gradients backprop(Gradients g, int miniBatchSize) {
        if(g.size() != 1){
            throw new IllegalArgumentException("Cannot preprocess activation gradients: Activation gradients must have " +
                    "exactly 1 array. Got: " + g.size());
        }
        INDArray output = g.get(0);
        if (output.ordering() == 'c')
            output = output.dup('f');

        int[] shape = output.shape();
        INDArray output2d;
        if (shape[0] == 1) {
            //Edge case: miniBatchSize = 1
            output2d = output.tensorAlongDimension(0, 1, 2).permutei(1, 0);
        } else if (shape[2] == 1) {
            //Edge case: timeSeriesLength = 1
            output2d = output.tensorAlongDimension(0, 1, 0);
        } else {
            //As per FeedForwardToRnnPreprocessor
            INDArray permuted3d = output.permute(0, 2, 1);
            output2d = permuted3d.reshape('f', shape[0] * shape[2], shape[1]);
        }

        if (shape[1] != product)
            throw new IllegalArgumentException("Invalid input: expected output size(1)=" + shape[1]
                            + " must be equal to " + inputHeight + " x columns " + inputWidth + " x depth "
                            + numChannels + " = " + product + ", received: " + shape[1]);
        INDArray ret = output2d.dup('c').reshape('c', output2d.size(0), numChannels, inputHeight, inputWidth);
        return GradientsFactory.getInstance().create(ret, g.getParameterGradients());
    }

    @Override
    public CnnToRnnPreProcessor clone() {
        return new CnnToRnnPreProcessor(inputHeight, inputWidth, numChannels);
    }

    @Override
    public InputType[] getOutputType(InputType... inputType) {
        if (inputType == null || inputType.length != 1 || inputType[0].getType() != InputType.Type.CNN) {
            throw new IllegalStateException("Invalid input type: Expected input of type CNN, got "
                    + (inputType == null ? null : Arrays.toString(inputType)));
        }

        InputType.InputTypeConvolutional c = (InputType.InputTypeConvolutional) inputType[0];
        int outSize = c.getDepth() * c.getHeight() * c.getWidth();
        return new InputType[]{InputType.recurrent(outSize)};
    }


    public Pair<INDArray, MaskState> feedForwardMaskArray(INDArray maskArray, MaskState currentMaskState,
                    int minibatchSize) {
        //Assume mask array is 1d - a mask array that has been reshaped from [minibatch,timeSeriesLength] to [minibatch*timeSeriesLength, 1]
        if (maskArray == null) {
            return new Pair<>(maskArray, currentMaskState);
        } else if (maskArray.isVector()) {
            //Need to reshape mask array from [minibatch*timeSeriesLength, 1] to [minibatch,timeSeriesLength]
            return new Pair<>(TimeSeriesUtils.reshapeVectorToTimeSeriesMask(maskArray, minibatchSize),
                            currentMaskState);
        } else {
            throw new IllegalArgumentException("Received mask array with shape " + Arrays.toString(maskArray.shape())
                            + "; expected vector.");
        }
    }
}
