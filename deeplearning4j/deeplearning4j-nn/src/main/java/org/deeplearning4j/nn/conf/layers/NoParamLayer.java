package org.deeplearning4j.nn.conf.layers;

import lombok.NoArgsConstructor;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.params.EmptyParamInitializer;

@NoArgsConstructor
public abstract class NoParamLayer extends Layer {

    protected NoParamLayer(Builder builder){
        super(builder);
    }

    @Override
    public ParamInitializer initializer() {
        return EmptyParamInitializer.getInstance();
    }

    @Override
    public void setNIn(InputType inputType, boolean override) {
        //No op in most no param layers
    }

    @Override
    public double getL1ByParam(String paramName) {
        return 0.0;
    }

    @Override
    public double getL2ByParam(String paramName) {
        return 0.0;
    }

    @Override
    public boolean isPretrainParam(String paramName) {
        throw new UnsupportedOperationException(getClass().getSimpleName() + " does not contain parameters");
    }
}
