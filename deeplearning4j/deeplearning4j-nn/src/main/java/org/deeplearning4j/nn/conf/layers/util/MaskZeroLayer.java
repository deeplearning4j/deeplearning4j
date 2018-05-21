package org.deeplearning4j.nn.conf.layers.util;

import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.wrapper.BaseWrapperLayer;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;

/*
 Wrapper which masks timesteps with 0 activation.
 Assumes that the input shape is [batch_size, input_size, timesteps].
 @author Martin Boyanov mboyanov@gmail.com
 */
public class MaskZeroLayer extends BaseWrapperLayer {


    /**
     *
     */
    private static final long serialVersionUID = 9074525846200921839L;

    public MaskZeroLayer(Layer underlying) {
        this.underlying = underlying;
    }

    @Override
    public org.deeplearning4j.nn.api.Layer instantiate(NeuralNetConfiguration conf, Collection<TrainingListener> trainingListeners,
                                                       int layerIndex, INDArray layerParamsView, boolean initializeParams) {

        NeuralNetConfiguration conf2 = conf.clone();
        conf2.setLayer(((BaseWrapperLayer)conf2.getLayer()).getUnderlying());

        org.deeplearning4j.nn.api.Layer underlyingLayer = underlying.instantiate(conf2, trainingListeners, layerIndex, layerParamsView, initializeParams);
        return new org.deeplearning4j.nn.layers.recurrent.MaskZeroLayer(underlyingLayer);
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        return underlying.getOutputType(layerIndex, inputType);
    }

    @Override
    public void setNIn(InputType inputType, boolean override) {
        underlying.setNIn(inputType, override);
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        return underlying.getPreProcessorForInputType(inputType);    //No op
    }

    @Override
    public double getL1ByParam(String paramName) {
        return underlying.getL1ByParam(paramName);   //No params
    }

    @Override
    public double getL2ByParam(String paramName) {
        return underlying.getL2ByParam(paramName);   //No params
    }

    @Override
    public boolean isPretrainParam(String paramName) {
        return false;
    }

    @Override
    public LayerMemoryReport getMemoryReport(InputType inputType) {
        return underlying.getMemoryReport(inputType);
    }

    @Override
    public String toString(){
        return "MaskZeroLayer(" + underlying.toString() + ")";
    }

}