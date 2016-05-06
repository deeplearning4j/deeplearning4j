package org.deeplearning4j.nn.conf.preprocessor;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AccessLevel;
import lombok.Data;
import lombok.Getter;
import lombok.Setter;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.util.ArrayUtil;

/**
 * A preprocessor to allow RNN and CNN layers to be used together<br>
 * For example, time series (video) input -> ConvolutionLayer, or conceivable GravesLSTM -> ConvolutionLayer<br>
 * Functionally equivalent to combining RnnToFeedForwardPreProcessor + FeedForwardToCnnPreProcessor<br>
 * Specifically, this does two things:<br>
 * (a) Reshape 3d activations out of RNN layer, with shape [miniBatchSize, numChannels*inputHeight*inputWidth, timeSeriesLength])
 * into 4d (CNN) activations (with shape [numExamples*timeSeriesLength, numChannels, inputWidth, inputHeight]) <br>
 * (b) Reshapes 4d epsilons (weights.*deltas) out of CNN layer (with shape
 * [numExamples*timeSeriesLength, numChannels, inputHeight, inputWidth]) into 3d epsilons with shape
 * [miniBatchSize, numChannels*inputHeight*inputWidth, timeSeriesLength] suitable to feed into CNN layers.
 * Note: numChannels is equivalent to depth or featureMaps referenced in different literature
 *
 * @author Alex Black
 */
@Data
public class RnnToCnnPreProcessor implements InputPreProcessor {

    private int inputHeight;
    private int inputWidth;
    private int numChannels;

    @Getter(AccessLevel.NONE)
    @Setter(AccessLevel.NONE)
    private int product;

    public RnnToCnnPreProcessor(@JsonProperty("inputHeight") int inputHeight,
                                @JsonProperty("inputWidth") int inputWidth,
                                @JsonProperty("numChannels") int numChannels) {
        this.inputHeight = inputHeight;
        this.inputWidth = inputWidth;
        this.numChannels = numChannels;
        this.product = inputHeight * inputWidth * numChannels;
    }


    @Override
    public INDArray preProcess(INDArray input, int miniBatchSize) {
        if(input.ordering() == 'c') input = input.dup('f');
        //Input: 3d activations (RNN)
        //Output: 4d activations (CNN)
        int[] shape = input.shape();
        INDArray in2d;
        if (shape[0] == 1) {
            //Edge case: miniBatchSize = 1
            in2d = input.tensorAlongDimension(0, 1, 2).permutei(1,0);
        } else if (shape[2] == 1) {
            //Edge case: time series length = 1
            in2d = input.tensorAlongDimension(0, 1, 0);
        } else {
            INDArray permuted = input.permute(0, 2, 1);    //Permute, so we get correct order after reshaping
            in2d = permuted.reshape('f',shape[0] * shape[2], shape[1]);
        }

        return in2d.dup('c').reshape('c',shape[0] * shape[2], numChannels, inputHeight, inputWidth);
    }

    @Override
    public INDArray backprop(INDArray output, int miniBatchSize) {
        //Input: 4d epsilons (CNN)
        //Output: 3d epsilons (RNN)
        if(output.ordering() == 'f') output = output.dup('c');
        int[] shape = output.shape();
        //First: reshape 4d to 2d
        INDArray twod = output.reshape('c',output.size(0), ArrayUtil.prod(output.shape())/output.size(0));
        //Second: reshape 2d to 4d
        INDArray reshaped = twod.dup('f').reshape('f',miniBatchSize,shape[0]/miniBatchSize,product);
        return reshaped.permute(0,2,1);
    }

    @Override
    public RnnToCnnPreProcessor clone() {
        return new RnnToCnnPreProcessor(inputHeight, inputWidth, numChannels);
    }
}
