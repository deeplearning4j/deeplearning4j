package org.deeplearning4j.nn.modelimport.keras.preprocessors;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.shade.jackson.annotation.JsonCreator;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Specialized CnnToFeedForwardInputPreProcessor for use with
 * Convolutional layers imported from Keras using the TensorFlow
 * backend.
 *
 * @author dave@skymind.io
 */
@Slf4j
public class TensorFlowCnnToFeedForwardPreProcessor extends CnnToFeedForwardPreProcessor {

    @JsonCreator
    public TensorFlowCnnToFeedForwardPreProcessor(@JsonProperty("inputHeight") int inputHeight,
                                                  @JsonProperty("inputWidth") int inputWidth,
                                                  @JsonProperty("numChannels") int numChannels) {
        super(inputHeight, inputWidth, numChannels);
    }

    public TensorFlowCnnToFeedForwardPreProcessor(int inputHeight, int inputWidth) {
        super(inputHeight, inputWidth);
    }

    public TensorFlowCnnToFeedForwardPreProcessor() {
        super();
    }

    @Override
    public INDArray preProcess(INDArray input, int miniBatchSize) {
        if(input.rank() == 2) return input; //Should usually never happen
        /* DL4J convolutional input:       # channels, # rows, # cols
         * TensorFlow convolutional input: # rows, # cols, # channels
         * Theano convolutional input:     # channels, # rows, # cols
         */

        /* TODO: remove the extra copies of the input. These are only
         * used for debugging purposes during development and testing.
         */
        INDArray flatInput = super.preProcess(input, miniBatchSize);
        INDArray permuted = input.permute(0, 2, 3, 1);
        INDArray flatPermuted = super.preProcess(permuted, miniBatchSize);
        return flatPermuted;
    }

    @Override
    public INDArray backprop(INDArray epsilons, int miniBatchSize) {
        INDArray epsilonsReshaped = super.backprop(epsilons, miniBatchSize);
        return epsilonsReshaped.permute(0, 3, 1, 2);
    }

    @Override
    public TensorFlowCnnToFeedForwardPreProcessor clone() {
        return (TensorFlowCnnToFeedForwardPreProcessor) super.clone();
    }
}
