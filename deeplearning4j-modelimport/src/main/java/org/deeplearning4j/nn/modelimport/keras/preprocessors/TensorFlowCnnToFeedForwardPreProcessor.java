package org.deeplearning4j.nn.modelimport.keras.preprocessors;

import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.shade.jackson.annotation.JsonCreator;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Created by davekale on 1/5/17.
 */
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

        INDArray inputPermuted = input.permute(0, 2, 3, 1);
        return super.preProcess(inputPermuted, miniBatchSize);
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
