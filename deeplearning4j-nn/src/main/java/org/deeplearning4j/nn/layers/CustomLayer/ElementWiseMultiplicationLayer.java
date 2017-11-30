package org.deeplearning4j.nn.layers.CustomLayer;

import Utils.Utils_dpl4j.CustomParamInitializer.ElementWiseParamInitializer;
import lombok.*;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;
import java.util.Map;


/**
 * created by jingshu
 */
@Data
@NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public class ElementWiseMultiplicationLayer extends org.deeplearning4j.nn.conf.layers.FeedForwardLayer {

//  ATTENTION!!! we have to add an empty constructor for custom layers otherwise we will have errors when loading the model
    public ElementWiseMultiplicationLayer(){}

    protected ElementWiseMultiplicationLayer(Builder builder){
        super(builder);
    }

    @Override
    public ElementWiseMultiplicationLayer clone() {
        ElementWiseMultiplicationLayer clone = (ElementWiseMultiplicationLayer) super.clone();
        return clone;
    }

    @Override
    public Layer instantiate(NeuralNetConfiguration conf, Collection<IterationListener> iterationListeners, int layerIndex, INDArray layerParamsView, boolean initializeParams) {
        if(this.nIn!=this.nOut) throw  new IllegalStateException("element wise layer must have the same input and output size!!!");
        ElementWiseMultiplicationLayer_impl ret = new ElementWiseMultiplicationLayer_impl(conf);
        ret.setListeners(iterationListeners);
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

        int numParams = initializer().numParams(this);
        int updaterStateSize = (int) getIUpdater().stateSize(numParams);

        int trainSizeFixed = 0;
        int trainSizeVariable = 0;
        if (getDropOut() > 0) {
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
