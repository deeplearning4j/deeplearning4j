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

package org.deeplearning4j.nn.conf.distribution;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.nd4j.shade.jackson.annotation.JsonCreator;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Orthogonal distribution, with gain parameter.<br>
 * See <a href="http://arxiv.org/abs/1312.6120">http://arxiv.org/abs/1312.6120</a> for details
 *
 */
@EqualsAndHashCode(callSuper = false)
@Data
public class OrthogonalDistribution extends Distribution {

    private double gain;

    /**
     * Create a log-normal distribution
     * with the given mean and std
     *
     * @param gain the gain
     */
    @JsonCreator
    public OrthogonalDistribution(@JsonProperty("gain") double gain) {
        this.gain = gain;
    }

    public String toString() {
        return "OrthogonalDistribution{gain=" + gain + "}";
    }
}
