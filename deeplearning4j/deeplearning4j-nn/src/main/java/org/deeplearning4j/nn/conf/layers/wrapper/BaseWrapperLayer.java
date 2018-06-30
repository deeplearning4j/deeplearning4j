package org.deeplearning4j.nn.conf.layers.wrapper;

import lombok.Data;
import lombok.NonNull;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.BaseLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.params.WrapperLayerParamInitializer;

/**
 * Base wrapper layer: the idea is to pass through all methods to the underlying layer, and selectively override
 * them as required. This is to save implementing every single passthrough method for all 'wrapper' layer subtypes
 *
 * @author Alex Black
 */
@Data
public abstract class BaseWrapperLayer extends Layer {

    protected Layer underlying;

    protected BaseWrapperLayer(){ }

    public BaseWrapperLayer(Layer underlying){
        this.underlying = underlying;
    }

    @Override
    public ParamInitializer initializer() {
        return WrapperLayerParamInitializer.getInstance();
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
    public GradientNormalization getGradientNormalization(String param) {
        return underlying.getGradientNormalization(param);
    }

    @Override
    public double getGradientNormalizationThreshold() {
        return underlying.getGradientNormalizationThreshold();
    }

    @Override
    public boolean isPretrainParam(String paramName) {
        return underlying.isPretrainParam(paramName);
    }

    @Override
    public LayerMemoryReport getMemoryReport(InputType inputType) {
        return underlying.getMemoryReport(inputType);
    }

    @Override
    public void setLayerName(String layerName){
        super.setLayerName(layerName);
        if(underlying != null){
            //May be null at some points during JSON deserialization
            underlying.setLayerName(layerName);
        }
    }

    @Override
    public boolean isPretrain() {
        return underlying.isPretrain();
    }
}
