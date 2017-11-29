package org.deeplearning4j.arbiter.layers;

import lombok.*;
import org.deeplearning4j.arbiter.dropout.DropoutSpace;
import org.deeplearning4j.arbiter.optimize.api.AbstractParameterSpace;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.FixedValue;
import org.deeplearning4j.nn.conf.dropout.IDropout;
import org.deeplearning4j.nn.conf.layers.DropoutLayer;

import java.util.Collections;
import java.util.List;

@Data
@EqualsAndHashCode(callSuper = true)
@NoArgsConstructor(access = AccessLevel.PROTECTED) //For Jackson JSON/YAML deserialization
public class DropoutLayerSpace extends AbstractParameterSpace<DropoutLayer> {

    protected ParameterSpace<IDropout> dropout;

    public DropoutLayerSpace(@NonNull ParameterSpace<IDropout> dropout){
        this.dropout = dropout;
    }

    protected DropoutLayerSpace(Builder builder){
        this(builder.dropout);
    }

    @Override
    public DropoutLayer getValue(double[] parameterValues) {
        return new DropoutLayer.Builder().dropOut(dropout.getValue(parameterValues)).build();
    }

    @Override
    public int numParameters() {
        return dropout.numParameters();
    }

    @Override
    public List<ParameterSpace> collectLeaves() {
        return Collections.<ParameterSpace>singletonList(dropout);
    }

    @Override
    public boolean isLeaf() {
        return false;
    }

    @Override
    public void setIndices(int... indices) {
        dropout.setIndices(indices);
    }

    public static class Builder {

        private ParameterSpace<IDropout> dropout;

        public Builder dropOut(double d){
            return iDropOut(new DropoutSpace(new FixedValue<>(d)));
        }

        public Builder dropOut(ParameterSpace<Double> dropOut){
            return iDropOut(new DropoutSpace(dropOut));
        }

        public Builder iDropOut(ParameterSpace<IDropout> dropout){
            this.dropout = dropout;
            return this;
        }

        public DropoutLayerSpace build(){
            return new DropoutLayerSpace(this);
        }
    }
}
