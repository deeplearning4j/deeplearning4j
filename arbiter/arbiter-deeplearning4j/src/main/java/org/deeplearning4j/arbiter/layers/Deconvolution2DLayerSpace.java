package org.deeplearning4j.arbiter.layers;


import lombok.AccessLevel;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import org.deeplearning4j.nn.conf.layers.Deconvolution2D;

@Data
@EqualsAndHashCode(callSuper = true)
@NoArgsConstructor(access = AccessLevel.PROTECTED) //For Jackson JSON/YAML deserialization
public class Deconvolution2DLayerSpace extends BaseConvolutionLayerSpace<Deconvolution2D> {

    protected Deconvolution2DLayerSpace(Builder builder){
        super(builder);
    }

    @Override
    public Deconvolution2D getValue(double[] parameterValues) {
        Deconvolution2D.Builder b = new Deconvolution2D.Builder();
        setLayerOptionsBuilder(b, parameterValues);
        return b.build();
    }

    public static class Builder extends BaseConvolutionLayerSpace.Builder<Builder> {


        @Override
        public Deconvolution2DLayerSpace build() {
            return new Deconvolution2DLayerSpace(this);
        }
    }
}
