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

package org.deeplearning4j.nn.api.layers;

import org.deeplearning4j.nn.api.Layer;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;

import java.io.Serializable;
import java.util.Set;

@JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, include = JsonTypeInfo.As.PROPERTY, property = "@class")
public interface LayerConstraint extends Cloneable, Serializable {

    /**
     * Apply a given constraint to a layer at each iteration
     * in the provided epoch, after parameters have been updated.
     *
     * @param layer org.deeplearning4j.nn.api.Layer
     * @param iteration given iteration as integer
     * @param epoch current epoch as integer
     */
    void applyConstraint(Layer layer, int iteration, int epoch);

    /**
     * Set the parameters that this layer constraint should be applied to
     *
     * @param params Parameters that the layer constraint should be applied to
     */
    void setParams(Set<String> params);

    /**
     * @return Set of parameters that this layer constraint will be applied to
     */
    Set<String> getParams();

    LayerConstraint clone();

}
