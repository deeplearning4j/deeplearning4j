package org.deeplearning4j.arbiter.layers;

import lombok.AccessLevel;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.FixedValue;
import org.deeplearning4j.nn.conf.layers.SeparableConvolution2D;

@Data
@EqualsAndHashCode(callSuper = true)
@NoArgsConstructor(access = AccessLevel.PROTECTED) //For Jackson JSON/YAML deserialization
public class SeparableConvolution2DLayerSpace extends BaseConvolutionLayerSpace<SeparableConvolution2D> {

    private ParameterSpace<Integer> depthMultiplier;

    protected SeparableConvolution2DLayerSpace(Builder builder){
        super(builder);
        this.depthMultiplier = builder.depthMultiplier;
    }

    @Override
    public SeparableConvolution2D getValue(double[] parameterValues) {
        SeparableConvolution2D.Builder b = new SeparableConvolution2D.Builder();
        setLayerOptionsBuilder(b, parameterValues);
        return b.build();
    }

    protected void setLayerOptionsBuilder(SeparableConvolution2D.Builder builder, double[] values){
        super.setLayerOptionsBuilder(builder, values);
        if (kernelSize != null)
            builder.kernelSize(kernelSize.getValue(values));
        if (stride != null)
            builder.stride(stride.getValue(values));
        if (padding != null)
            builder.padding(padding.getValue(values));
        if (convolutionMode != null)
            builder.convolutionMode(convolutionMode.getValue(values));
        if (hasBias != null)
            builder.hasBias(hasBias.getValue(values));
        if (depthMultiplier != null)
            builder.depthMultiplier(depthMultiplier.getValue(values));
    }


    public static class Builder extends BaseConvolutionLayerSpace.Builder<Builder>{
        private ParameterSpace<Integer> depthMultiplier;

        public Builder depthMultiplier(int depthMultiplier){
            return depthMultiplier(new FixedValue<>(depthMultiplier));
        }

        public Builder depthMultiplier(ParameterSpace<Integer> depthMultiplier){
            this.depthMultiplier = depthMultiplier;
            return this;
        }

        public SeparableConvolution2DLayerSpace build(){
            return new SeparableConvolution2DLayerSpace(this);
        }
    }
}
