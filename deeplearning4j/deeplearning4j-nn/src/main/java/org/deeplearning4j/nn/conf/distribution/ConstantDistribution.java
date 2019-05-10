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
 * Constant distribution: a "distribution" where all values are set to the specified constant
 *
 */
@Data
@EqualsAndHashCode(callSuper = false)
public class ConstantDistribution extends Distribution {

    private double value;

    /**
     * Create a Constant distribution with given value
     *
     * @param value the gain
     */
    @JsonCreator
    public ConstantDistribution(@JsonProperty("value") double value) {
        this.value = value;
    }

    public String toString() {
        return "ConstantDistribution(value=" + value + ")";
    }
}
