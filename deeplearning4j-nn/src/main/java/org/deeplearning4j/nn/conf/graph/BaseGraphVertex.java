package org.deeplearning4j.nn.conf.graph;

import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.params.EmptyParamInitializer;

public abstract class BaseGraphVertex extends Layer {

    @Override
    public ParamInitializer initializer() {
        return EmptyParamInitializer.getInstance();
    }

    @Override
    public void setNIn(InputType[] inputType, boolean override) {
        //No op
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType... inputType) {
        return null;
    }

    @Override
    public double getL1ByParam(String paramName) {
        //No params
        return 0;
    }

    @Override
    public double getL2ByParam(String paramName) {
        //No params
        return 0;
    }

    @Override
    public boolean isPretrainParam(String paramName) {
        //No params
        return false;
    }
}
