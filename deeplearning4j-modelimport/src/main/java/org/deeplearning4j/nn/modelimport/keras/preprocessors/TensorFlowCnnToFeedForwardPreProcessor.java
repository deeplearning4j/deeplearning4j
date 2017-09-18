package org.deeplearning4j.nn.modelimport.keras.preprocessors;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.activations.Activations;
import org.deeplearning4j.nn.api.activations.ActivationsFactory;
import org.deeplearning4j.nn.api.gradients.Gradients;
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
    public Activations preProcess(Activations a, int miniBatchSize) {
        if(a.size() != 1){
            throw new IllegalArgumentException("Cannot preprocess input: Activations must have exactly 1 array. Got: "
                    + a.size());
        }
        INDArray input = a.get(0);
        if (input.rank() == 2)
            return a; //Should usually never happen
        /* DL4J convolutional input:       # channels, # rows, # cols
         * TensorFlow convolutional input: # rows, # cols, # channels
         * Theano convolutional input:     # channels, # rows, # cols
         */
        INDArray permuted = input.permute(0, 2, 3, 1);
        Activations a2 = ActivationsFactory.getInstance().create(permuted, a.getMask(0), a.getMaskState(0));
        return super.preProcess(a2, miniBatchSize);
    }

    @Override
    public Gradients backprop(Gradients g, int miniBatchSize) {
        Gradients gReshaped = super.backprop(g, miniBatchSize);
        gReshaped.set(0, gReshaped.get(0).permute(0, 3, 1, 2));
        return gReshaped;
    }

    @Override
    public TensorFlowCnnToFeedForwardPreProcessor clone() {
        return (TensorFlowCnnToFeedForwardPreProcessor) super.clone();
    }
}
