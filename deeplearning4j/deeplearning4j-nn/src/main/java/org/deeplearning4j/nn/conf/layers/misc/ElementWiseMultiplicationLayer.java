package org.deeplearning4j.nn.conf.layers.misc;

import lombok.*;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.params.ElementWiseParamInitializer;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;
import java.util.Map;


/**
 * Elementwise multiplication layer with weights: implements out = activationFn(input .* w + b) where:<br>
 * - w is a learnable weight vector of length nOut<br>
 * - ".*" is element-wise multiplication<br>
 * - b is a bias vector<br>
 * <br>
 * Note that the input and output sizes of the element-wise layer are the same for this layer
 * <p>
 * created by jingshu
 */
@Data
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public class ElementWiseMultiplicationLayer extends org.deeplearning4j.nn.conf.layers.FeedForwardLayer {

    //  We have to add an empty constructor for custom layers otherwise we will have errors when loading the model
    protected ElementWiseMultiplicationLayer() { }

    protected ElementWiseMultiplicationLayer(Builder builder) {
        super(builder);
    }

    @Override
    public ElementWiseMultiplicationLayer clone() {
        ElementWiseMultiplicationLayer clone = (ElementWiseMultiplicationLayer) super.clone();
        return clone;
    }

    @Override
    public Layer instantiate(NeuralNetConfiguration conf, Collection<TrainingListener> trainingListeners, int layerIndex,
                             INDArray layerParamsView, boolean initializeParams) {
        if (this.nIn != this.nOut) {
            throw new IllegalStateException("Element wise layer must have the same input and output size. Got nIn="
                    + nIn + ", nOut=" + nOut);
        }
        org.deeplearning4j.nn.layers.feedforward.elementwise.ElementWiseMultiplicationLayer ret
                = new org.deeplearning4j.nn.layers.feedforward.elementwise.ElementWiseMultiplicationLayer(conf);
        ret.setListeners(trainingListeners);
        ret.setIndex(layerIndex);
        ret.setParamsViewArray(layerParamsView);
        Map<String, INDArray> paramTable = initializer().init(conf, layerParamsView, initializeParams);
        ret.setParamTable(paramTable);
        ret.setConf(conf);

        return ret;
    }

    @Override
    public ParamInitializer initializer() {
        return ElementWiseParamInitializer.getInstance();
    }

    /**
     * This is a report of the estimated memory consumption for the given layer
     *
     * @param inputType Input type to the layer. Memory consumption is often a function of the input type
     * @return Memory report for the layer
     */
    @Override
    public LayerMemoryReport getMemoryReport(InputType inputType) {
        InputType outputType = getOutputType(-1, inputType);

        val numParams = initializer().numParams(this);
        val updaterStateSize = (int) getIUpdater().stateSize(numParams);

        int trainSizeFixed = 0;
        int trainSizeVariable = 0;
        if (getIDropout() != null) {
            if (false) {
                //TODO drop connect
                //Dup the weights... note that this does NOT depend on the minibatch size...
                trainSizeVariable += 0; //TODO
            } else {
                //Assume we dup the input
                trainSizeVariable += inputType.arrayElementsPerExample();
            }
        }

        //Also, during backprop: we do a preOut call -> gives us activations size equal to the output size
        // which is modified in-place by activation function backprop
        // then we have 'epsilonNext' which is equivalent to input size
        trainSizeVariable += outputType.arrayElementsPerExample();

        return new LayerMemoryReport.Builder(layerName, ElementWiseMultiplicationLayer.class, inputType, outputType)
                .standardMemory(numParams, updaterStateSize)
                .workingMemory(0, 0, trainSizeFixed, trainSizeVariable) //No additional memory (beyond activations) for inference
                .cacheMemory(MemoryReport.CACHE_MODE_ALL_ZEROS, MemoryReport.CACHE_MODE_ALL_ZEROS) //No caching in DenseLayer
                .build();
    }


    @AllArgsConstructor
    public static class Builder extends FeedForwardLayer.Builder<ElementWiseMultiplicationLayer.Builder> {

        @Override
        @SuppressWarnings("unchecked")
        public ElementWiseMultiplicationLayer build() {
            return new ElementWiseMultiplicationLayer(this);
        }
    }
}
