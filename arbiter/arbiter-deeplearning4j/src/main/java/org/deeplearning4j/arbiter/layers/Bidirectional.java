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

import lombok.Data;
import lombok.NoArgsConstructor;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.nn.conf.layers.Layer;

import java.util.List;

/**
 * Bidirectional layer wrapper. Can be used wrap an existing layer space, in the same way that
 * {@link org.deeplearning4j.nn.conf.layers.recurrent.Bidirectional} wraps a DL4J layer
 *
 * @author Alex Black
 */
@NoArgsConstructor  //JSON
@Data
public class Bidirectional extends LayerSpace<Layer> {

    protected LayerSpace<?> layerSpace;

    public Bidirectional(LayerSpace<?> layerSpace){
        this.layerSpace = layerSpace;
    }

    @Override
    public Layer getValue(double[] parameterValues) {
        Layer underlying = layerSpace.getValue(parameterValues);
        return new org.deeplearning4j.nn.conf.layers.recurrent.Bidirectional(underlying);
    }

    @Override
    public int numParameters() {
        return layerSpace.numParameters();
    }

    @Override
    public List<ParameterSpace> collectLeaves() {
        return layerSpace.collectLeaves();
    }

    @Override
    public boolean isLeaf() {
        return layerSpace.isLeaf();
    }

    @Override
    public void setIndices(int... indices) {
        layerSpace.setIndices(indices);
    }
}
