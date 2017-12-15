package org.deeplearning4j.nn.conf.layers.wrapper;

import lombok.NonNull;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;

/**
 * Base wrapper layer
 *
 */
public abstract class BaseWrapperLayer extends Layer {

    protected Layer underlying;

    public BaseWrapperLayer(@NonNull Layer underlying){
        this.underlying = underlying;
    }

    @Override
    public ParamInitializer initializer() {
        
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {

    }

    @Override
    public void setNIn(InputType inputType, boolean override) {
        underlying.setNIn(inputType, override);
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        return underlying.getPreProcessorForInputType(inputType);
    }

    @Override
    public double getL1ByParam(String paramName) {
        return underlying.getL1ByParam(paramName);
    }

    @Override
    public double getL2ByParam(String paramName) {
        return underlying.getL2ByParam(paramName);
    }

    @Override
    public boolean isPretrainParam(String paramName) {
        return underlying.isPretrainParam(paramName);
    }

    @Override
    public LayerMemoryReport getMemoryReport(InputType inputType) {
        return underlying.getMemoryReport(inputType);
    }
}
