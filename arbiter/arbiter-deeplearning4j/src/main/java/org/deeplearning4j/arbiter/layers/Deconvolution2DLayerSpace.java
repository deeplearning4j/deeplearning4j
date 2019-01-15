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
import org.deeplearning4j.nn.conf.layers.convolutional.Deconvolution2DLayer;

@Data
@EqualsAndHashCode(callSuper = true)
@NoArgsConstructor(access = AccessLevel.PROTECTED) //For Jackson JSON/YAML deserialization
public class Deconvolution2DLayerSpace extends BaseConvolutionLayerSpace<Deconvolution2DLayer> {

    protected Deconvolution2DLayerSpace(Builder builder) {
        super(builder);
    }

    @Override
    public Deconvolution2DLayer getValue(double[] parameterValues) {
        Deconvolution2DLayer.Builder b = new Deconvolution2DLayer.Builder();
        setLayerOptionsBuilder(b, parameterValues);
        return b.build();
    }

    protected void setLayerOptionsBuilder(Deconvolution2DLayer.Builder builder, double[] values) {
        super.setLayerOptionsBuilder(builder, values);
    }

    public static class Builder extends BaseConvolutionLayerSpace.Builder<Builder> {
        @Override
        public Deconvolution2DLayerSpace build() {
            return new Deconvolution2DLayerSpace(this);
        }
    }
}
