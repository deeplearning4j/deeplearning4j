package org.deeplearning4j.nn.conf.layers.samediff;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.params.EmptyParamInitializer;
import org.deeplearning4j.nn.params.SameDiffParamInitializer;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.config.IUpdater;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;

@Data
@EqualsAndHashCode(callSuper = true)
public abstract class NoParamSameDiffLayer extends Layer {

    protected NoParamSameDiffLayer(Builder builder){
        super(builder);
    }

    protected NoParamSameDiffLayer(){
        //No op constructor for Jackson
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType){
        return inputType;
    }

    @Override
    public void setNIn(InputType inputType, boolean override){
        //No op
    }

    @Override
    public abstract InputPreProcessor getPreProcessorForInputType(InputType inputType);

    public abstract void applyGlobalConfigToLayer(NeuralNetConfiguration.Builder globalConfig);

    @Override
    public abstract org.deeplearning4j.nn.api.Layer instantiate(NeuralNetConfiguration conf, Collection<IterationListener> iterationListeners,
                                                                int layerIndex, INDArray layerParamsView, boolean initializeParams);

    //==================================================================================================================

    @Override
    public ParamInitializer initializer() {
        return EmptyParamInitializer.getInstance();
    }

    @Override
    public double getL1ByParam(String paramName) {
        return 0.0; //No params
    }

    @Override
    public double getL2ByParam(String paramName) {
        return 0.0; //No params
    }

    @Override
    public IUpdater getUpdaterByParam(String paramName){
        throw new UnsupportedOperationException("No parameters for this layer");
    }

    @Override
    public boolean isPretrainParam(String paramName) {
        return false;
    }

    @Override
    public LayerMemoryReport getMemoryReport(InputType inputType) {
        return new LayerMemoryReport(); //TODO
    }

    public void applyGlobalConfig(NeuralNetConfiguration.Builder b){
        applyGlobalConfigToLayer(b);
    }

    public static abstract class Builder<T extends Builder<T>> extends Layer.Builder<T> {

    }
}
