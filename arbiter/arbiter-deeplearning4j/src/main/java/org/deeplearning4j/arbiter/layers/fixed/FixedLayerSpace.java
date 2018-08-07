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

package org.deeplearning4j.arbiter.layers.fixed;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import org.deeplearning4j.arbiter.layers.LayerSpace;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.nn.conf.layers.Layer;

import java.util.Collections;
import java.util.List;

/**
 * A layer space that wraps a DL4J layer, without any optimizable hyperparameters
 *
 * @param <T> Type of layer
 *
 * @author Alex Black
 */
@AllArgsConstructor
@NoArgsConstructor
@Data
@EqualsAndHashCode(callSuper = false)
public class FixedLayerSpace<T extends Layer> extends LayerSpace<T> {

    protected T layer;

    @Override
    public T getValue(double[] parameterValues) {
        return (T)layer.clone();
    }

    @Override
    public int numParameters() {
        return 0;
    }

    @Override
    public boolean isLeaf() {
        return true;
    }

    @Override
    public void setIndices(int[] idxs){
        if(idxs != null && idxs.length > 0){
            throw new IllegalStateException("Cannot set indices: no parameters");
        }
    }

    @Override
    public List<ParameterSpace> collectLeaves() {
        return Collections.<ParameterSpace>singletonList(this);
    }
}
