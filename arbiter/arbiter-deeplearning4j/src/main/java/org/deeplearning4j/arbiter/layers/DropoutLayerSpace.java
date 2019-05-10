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
public class DropoutLayerSpace extends LayerSpace<DropoutLayer> {

    public DropoutLayerSpace(@NonNull ParameterSpace<IDropout> dropout){
        this.dropOut = dropout;
    }

    protected DropoutLayerSpace(Builder builder){
        super(builder);
    }

    @Override
    public DropoutLayer getValue(double[] parameterValues) {
        return new DropoutLayer.Builder().dropOut(dropOut.getValue(parameterValues)).build();
    }

    @Override
    public int numParameters() {
        return dropOut.numParameters();
    }

    @Override
    public List<ParameterSpace> collectLeaves() {
        return Collections.<ParameterSpace>singletonList(dropOut);
    }

    @Override
    public boolean isLeaf() {
        return false;
    }

    @Override
    public void setIndices(int... indices) {
        dropOut.setIndices(indices);
    }

    public static class Builder extends LayerSpace.Builder<Builder> {

        public Builder dropOut(double d){
            return iDropOut(new DropoutSpace(new FixedValue<>(d)));
        }

        public Builder dropOut(ParameterSpace<Double> dropOut){
            return iDropOut(new DropoutSpace(dropOut));
        }

        public Builder iDropOut(ParameterSpace<IDropout> dropout){
            this.dropOut = dropout;
            return this;
        }

        public DropoutLayerSpace build(){
            return new DropoutLayerSpace(this);
        }
    }
}
