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
import org.deeplearning4j.nn.conf.layers.Deconvolution2D;

@Data
@EqualsAndHashCode(callSuper = true)
@NoArgsConstructor(access = AccessLevel.PROTECTED) //For Jackson JSON/YAML deserialization
public class Deconvolution2DLayerSpace extends BaseConvolutionLayerSpace<Deconvolution2D> {
    protected ParameterSpace<int[]> dilation;

    protected Deconvolution2DLayerSpace(Builder builder) {
        super(builder);
        this.dilation = builder.dilation;
    }

    @Override
    public Deconvolution2D getValue(double[] parameterValues) {
        Deconvolution2D.Builder b = new Deconvolution2D.Builder();
        setLayerOptionsBuilder(b, parameterValues);
        return b.build();
    }

    protected void setLayerOptionsBuilder(Deconvolution2D.Builder builder, double[] values) {
        super.setLayerOptionsBuilder(builder, values);
        if (dilation != null)
            builder.dilation(dilation.getValue(values));
    }

    public static class Builder extends BaseConvolutionLayerSpace.Builder<Builder> {
        protected ParameterSpace<int[]> dilation;

        public Builder dilation(int... dilation) {
            return dilation(new FixedValue<>(dilation));
        }

        public Builder dilation(ParameterSpace<int[]> dilation) {
            this.dilation = dilation;
            return this;
        }

        @Override
        public Deconvolution2DLayerSpace build() {
            return new Deconvolution2DLayerSpace(this);
        }
    }
}
