package org.deeplearning4j.arbiter.layers;

import lombok.AccessLevel;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.FixedValue;
import org.deeplearning4j.arbiter.util.LeafUtils;
import org.deeplearning4j.nn.conf.ocnn.OCNNOutputLayer;


@Data
@EqualsAndHashCode(callSuper = true)
@NoArgsConstructor(access = AccessLevel.PRIVATE) //For Jackson JSON/YAML deserialization
public class OCNNLayerSpace  extends BaseOutputLayerSpace<OCNNOutputLayer> {


    protected ParameterSpace<Double> nuSpace;
    protected ParameterSpace<Integer> numHiddenSpace;

    private OCNNLayerSpace(Builder builder) {
        super(builder);

        this.numParameters = LeafUtils.countUniqueParameters(collectLeaves());
        this.nuSpace = builder.nuSpace;
        this.numHiddenSpace = builder.numHiddenSpace;
    }



    @Override
    public OCNNOutputLayer getValue(double[] parameterValues) {
        OCNNOutputLayer.Builder o = new OCNNOutputLayer.Builder();
        setLayerOptionsBuilder(o, parameterValues);
        return o.build();
    }

    protected void setLayerOptionsBuilder(OCNNOutputLayer.Builder builder, double[] values) {
        super.setLayerOptionsBuilder(builder, values);
        builder.nu(nuSpace.getValue(values));
        builder.hiddenLayerSize(numHiddenSpace.getValue(values));
    }



    public  static class Builder extends BaseOutputLayerSpace.Builder<Builder> {
        protected ParameterSpace<Double> nuSpace;
        protected ParameterSpace<Integer> numHiddenSpace;

        public Builder nu(ParameterSpace<Double> nuSpace) {
            this.nuSpace = nuSpace;
            return this;
        }

        public Builder numHidden(ParameterSpace<Integer> numHiddenSpace) {
            this.numHiddenSpace = numHiddenSpace;
            return this;
        }

        public Builder numHidden(int numHidden) {
            this.numHiddenSpace = new FixedValue<>(numHidden);
            return this;
        }

        public Builder nu(double nu) {
            this.nuSpace = new FixedValue<>(nu);
            return this;
        }


        @Override
        @SuppressWarnings("unchecked")
        public OCNNLayerSpace build() {
            return new OCNNLayerSpace(this);
        }
    }
}
