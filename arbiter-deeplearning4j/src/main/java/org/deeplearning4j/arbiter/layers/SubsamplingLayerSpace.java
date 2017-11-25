/*-
 *
 *  * Copyright 2016 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */
package org.deeplearning4j.arbiter.layers;

import lombok.AccessLevel;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.FixedValue;
import org.deeplearning4j.arbiter.util.LeafUtils;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;

/**
 * Layer hyperparameter configuration space for subsampling layers
 *
 * @author Alex Black
 */
@Data
@EqualsAndHashCode(callSuper = true)
@NoArgsConstructor(access = AccessLevel.PRIVATE) //For Jackson JSON/YAML deserialization
public class SubsamplingLayerSpace extends LayerSpace<SubsamplingLayer> {

    protected ParameterSpace<ConvolutionMode> convolutionMode;
    protected ParameterSpace<SubsamplingLayer.PoolingType> poolingType;
    protected ParameterSpace<int[]> kernelSize;
    protected ParameterSpace<int[]> stride;
    protected ParameterSpace<int[]> padding;
    protected ParameterSpace<Integer> pnorm;
    protected ParameterSpace<Double> eps;

    private SubsamplingLayerSpace(Builder builder) {
        super(builder);
        this.convolutionMode = builder.convolutionMode;
        this.poolingType = builder.poolingType;
        this.kernelSize = builder.kernelSize;
        this.stride = builder.stride;
        this.padding = builder.padding;
        this.pnorm = builder.pnorm;
        this.eps = builder.eps;

        this.numParameters = LeafUtils.countUniqueParameters(collectLeaves());
    }

    @Override
    public SubsamplingLayer getValue(double[] values) {
        SubsamplingLayer.Builder b = new SubsamplingLayer.Builder();
        setLayerOptionsBuilder(b, values);
        return b.build();
    }

    protected void setLayerOptionsBuilder(SubsamplingLayer.Builder builder, double[] values) {
        super.setLayerOptionsBuilder(builder, values);
        if (convolutionMode != null)
            builder.convolutionMode(convolutionMode.getValue(values));
        if (poolingType != null)
            builder.poolingType(poolingType.getValue(values));
        if (kernelSize != null)
            builder.kernelSize(kernelSize.getValue(values));
        if (stride != null)
            builder.stride(stride.getValue(values));
        if (padding != null)
            builder.padding(padding.getValue(values));
        if(pnorm != null)
            builder.pnorm(pnorm.getValue(values));
        if(eps != null)
            builder.eps(eps.getValue(values));
    }

    @Override
    public String toString() {
        return toString(", ");
    }

    @Override
    public String toString(String delim) {
        StringBuilder sb = new StringBuilder("SubsamplingLayerSpace(");
        if (convolutionMode != null)
            sb.append("convolutionMode: ").append(convolutionMode).append(delim);
        if (poolingType != null)
            sb.append("poolingType: ").append(poolingType).append(delim);
        if (kernelSize != null)
            sb.append("kernelSize: ").append(kernelSize).append(delim);
        if (stride != null)
            sb.append("stride: ").append(stride).append(delim);
        if (padding != null)
            sb.append("padding: ").append(padding).append(delim);
        if (pnorm != null)
            sb.append("pnorm: ").append(pnorm).append(delim);
        if (eps != null)
            sb.append("eps: ").append(eps).append(delim);
        sb.append(super.toString(delim)).append(")");
        return sb.toString();
    }


    public static class Builder extends FeedForwardLayerSpace.Builder<Builder> {

        protected ParameterSpace<ConvolutionMode> convolutionMode;
        protected ParameterSpace<SubsamplingLayer.PoolingType> poolingType;
        protected ParameterSpace<int[]> kernelSize;
        protected ParameterSpace<int[]> stride;
        protected ParameterSpace<int[]> padding;
        protected ParameterSpace<Integer> pnorm;
        protected ParameterSpace<Double> eps;

        public Builder convolutionMode(ConvolutionMode convolutionMode){
            return convolutionMode(new FixedValue<>(convolutionMode));
        }

        public Builder convolutionMode(ParameterSpace<ConvolutionMode> convolutionMode){
            this.convolutionMode = convolutionMode;
            return this;
        }

        public Builder poolingType(SubsamplingLayer.PoolingType poolingType) {
            return poolingType(new FixedValue<>(poolingType));
        }

        public Builder poolingType(ParameterSpace<SubsamplingLayer.PoolingType> poolingType) {
            this.poolingType = poolingType;
            return this;
        }

        public Builder kernelSize(int... kernelSize) {
            return kernelSize(new FixedValue<>(kernelSize));
        }

        public Builder kernelSize(ParameterSpace<int[]> kernelSize) {
            this.kernelSize = kernelSize;
            return this;
        }

        public Builder stride(int... stride) {
            return stride(new FixedValue<int[]>(stride));
        }

        public Builder stride(ParameterSpace<int[]> stride) {
            this.stride = stride;
            return this;
        }

        public Builder padding(int... padding) {
            return padding(new FixedValue<int[]>(padding));
        }

        public Builder padding(ParameterSpace<int[]> padding) {
            this.padding = padding;
            return this;
        }

        public Builder pnorm(int pnorm){
            return pnorm(new FixedValue<>(pnorm));
        }

        public Builder pnorm(ParameterSpace<Integer> pnorm){
            this.pnorm = pnorm;
            return this;
        }

        public Builder eps(double eps){
            return eps(new FixedValue<>(eps));
        }

        public Builder eps(ParameterSpace<Double> eps){
            this.eps = eps;
            return this;
        }

        @SuppressWarnings("unchecked")
        public SubsamplingLayerSpace build() {
            return new SubsamplingLayerSpace(this);
        }

    }

}
