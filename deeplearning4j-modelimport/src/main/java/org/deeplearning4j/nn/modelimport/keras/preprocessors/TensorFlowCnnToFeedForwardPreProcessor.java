package org.deeplearning4j.nn.modelimport.keras.preprocessors;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
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
                    @JsonProperty("inputWidth") int inputWidth, @JsonProperty("numChannels") int numChannels) {
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
        if (input.rank() == 2)
            return input; //Should usually never happen
        /* DL4J convolutional input:       # channels, # rows, # cols
         * TensorFlow convolutional input: # rows, # cols, # channels
         * Theano convolutional input:     # channels, # rows, # cols
         */
        INDArray permuted = input.permute(0, 2, 3, 1).dup('c'); //To: [n, h, w, c]

        int[] inShape = input.shape(); //[miniBatch,depthOut,outH,outW]
        int[] outShape = new int[] {inShape[0], inShape[1] * inShape[2] * inShape[3]};

        return permuted.reshape('c', outShape);
    }

    @Override
    public INDArray backprop(INDArray epsilons, int miniBatchSize) {
        if (epsilons.ordering() != 'c' || !Shape.strideDescendingCAscendingF(epsilons))
            epsilons = epsilons.dup('c');

        INDArray epsilonsReshaped = epsilons.reshape('c', epsilons.size(0), inputHeight, inputWidth, numChannels);

        return epsilonsReshaped.permute(0, 3, 1, 2);    //To [n, c, h, w]
    }

    @Override
    public TensorFlowCnnToFeedForwardPreProcessor clone() {
        return (TensorFlowCnnToFeedForwardPreProcessor) super.clone();
    }
}
