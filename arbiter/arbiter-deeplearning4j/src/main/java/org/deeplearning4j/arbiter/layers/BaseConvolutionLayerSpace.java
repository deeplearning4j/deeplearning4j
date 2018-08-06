/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.arbiter.layers;

import lombok.AccessLevel;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.FixedValue;
import org.deeplearning4j.arbiter.util.LeafUtils;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;

/**
 * Layer space for convolutional layers
 *
 * @author Alex Black
 */
@Data
@EqualsAndHashCode(callSuper = true)
@NoArgsConstructor(access = AccessLevel.PROTECTED) //For Jackson JSON/YAML deserialization
public abstract class BaseConvolutionLayerSpace<T extends FeedForwardLayer> extends FeedForwardLayerSpace<T> {
    protected ParameterSpace<int[]> dilation;
    protected ParameterSpace<int[]> kernelSize;
    protected ParameterSpace<int[]> stride;
    protected ParameterSpace<int[]> padding;
    protected ParameterSpace<ConvolutionMode> convolutionMode;
    protected ParameterSpace<Boolean> hasBias;

    protected BaseConvolutionLayerSpace(Builder builder) {
        super(builder);
        this.dilation = builder.dilation;
        this.kernelSize = builder.kernelSize;
        this.stride = builder.stride;
        this.padding = builder.padding;
        this.convolutionMode = builder.convolutionMode;
        this.hasBias = builder.hasBias;

        this.numParameters = LeafUtils.countUniqueParameters(collectLeaves());
    }

    protected void setLayerOptionsBuilder(ConvolutionLayer.BaseConvBuilder<?> builder, double[] values) {
        super.setLayerOptionsBuilder(builder, values);
        if (dilation != null)
            builder.dilation(dilation.getValue(values));
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
    }

    @Override
    public String toString() {
        return toString(", ");
    }

    @Override
    public String toString(String delim) {
        StringBuilder sb = new StringBuilder();
        if (dilation != null)
            sb.append("dilation: ").append(dilation).append(delim);
        if (kernelSize != null)
            sb.append("kernelSize: ").append(kernelSize).append(delim);
        if (stride != null)
            sb.append("stride: ").append(stride).append(delim);
        if (padding != null)
            sb.append("padding: ").append(padding).append(delim);
        if (convolutionMode != null)
            sb.append("convolutionMode: ").append(convolutionMode).append(delim);
        if (hasBias != null)
            sb.append("hasBias: ").append(hasBias).append(delim);
        sb.append(super.toString(delim));
        return sb.toString();
    }


    public static abstract class Builder<T> extends FeedForwardLayerSpace.Builder<T> {
        protected ParameterSpace<int[]> dilation;
        protected ParameterSpace<int[]> kernelSize;
        protected ParameterSpace<int[]> stride;
        protected ParameterSpace<int[]> padding;
        protected ParameterSpace<ConvolutionMode> convolutionMode;
        protected ParameterSpace<Boolean> hasBias;

        public T dilation(int... dilation) {
            return dilation(new FixedValue<>(dilation));
        }

        public T dilation(ParameterSpace<int[]> dilation) {
            this.dilation = dilation;
            return (T) this;
        }
        public T kernelSize(int... kernelSize) {
            return kernelSize(new FixedValue<>(kernelSize));
        }

        public T kernelSize(ParameterSpace<int[]> kernelSize) {
            this.kernelSize = kernelSize;
            return (T)this;
        }

        public T stride(int... stride) {
            return stride(new FixedValue<>(stride));
        }

        public T stride(ParameterSpace<int[]> stride) {
            this.stride = stride;
            return (T)this;
        }

        public T padding(int... padding) {
            return padding(new FixedValue<>(padding));
        }

        public T padding(ParameterSpace<int[]> padding) {
            this.padding = padding;
            return (T)this;
        }

        public T convolutionMode(ConvolutionMode convolutionMode) {
            return convolutionMode(new FixedValue<>(convolutionMode));
        }

        public T convolutionMode(ParameterSpace<ConvolutionMode> convolutionMode) {
            this.convolutionMode = convolutionMode;
            return (T)this;
        }

        public T hasBias(boolean hasBias){
            return hasBias(new FixedValue<>(hasBias));
        }

        public T hasBias(ParameterSpace<Boolean> hasBias){
            this.hasBias = hasBias;
            return (T)this;
        }

    }

}
