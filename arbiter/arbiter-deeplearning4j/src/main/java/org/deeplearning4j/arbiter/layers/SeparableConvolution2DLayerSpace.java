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
import org.deeplearning4j.nn.api.layers.LayerConstraint;
import org.deeplearning4j.nn.conf.layers.SeparableConvolution2D;

import java.util.Arrays;
import java.util.List;

@Data
@EqualsAndHashCode(callSuper = true)
@NoArgsConstructor(access = AccessLevel.PROTECTED) //For Jackson JSON/YAML deserialization
public class SeparableConvolution2DLayerSpace extends BaseConvolutionLayerSpace<SeparableConvolution2D> {

    private ParameterSpace<Integer> depthMultiplier;
    protected ParameterSpace<List<LayerConstraint>> pointWiseConstraints;

    protected SeparableConvolution2DLayerSpace(Builder builder){
        super(builder);
        this.depthMultiplier = builder.depthMultiplier;
        this.pointWiseConstraints = builder.pointWiseConstraints;
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
        if (pointWiseConstraints != null){
            List<LayerConstraint> c = pointWiseConstraints.getValue(values);
            if(c != null){
                builder.constrainPointWise(c.toArray(new LayerConstraint[c.size()]));
            }
        }
    }


    public static class Builder extends BaseConvolutionLayerSpace.Builder<Builder>{
        private ParameterSpace<Integer> depthMultiplier;
        protected ParameterSpace<List<LayerConstraint>> pointWiseConstraints;

        public Builder constrainPointWise(LayerConstraint... constraints){
            return constrainPointWise(new FixedValue<List<LayerConstraint>>(Arrays.asList(constraints)));
        }

        public Builder constrainPointWise(ParameterSpace<List<LayerConstraint>> constraints){
            this.pointWiseConstraints = constraints;
            return this;
        }

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
