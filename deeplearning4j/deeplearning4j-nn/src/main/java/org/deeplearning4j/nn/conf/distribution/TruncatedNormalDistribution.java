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
 * A truncated normal distribution, with 2 parameters: mean and standard deviation<br>
 * This distribution is a standard normal/Gaussian distribtion, however values are "truncated" in the sense that any
 * values that fall outside the range [mean - 2 * stdev, mean + 2 * stdev] are re-sampled.
 */
@EqualsAndHashCode(callSuper = false)
@Data
public class TruncatedNormalDistribution extends Distribution {

    private double mean, std;

    /**
     * Create a truncated normal distribution
     * with the given mean and std
     *
     * @param mean the mean
     * @param std  the standard deviation
     */
    @JsonCreator
    public TruncatedNormalDistribution(@JsonProperty("mean") double mean, @JsonProperty("std") double std) {
        this.mean = mean;
        this.std = std;
    }

    public String toString() {
        return "TruncatedNormalDistribution(" + "mean=" + mean + ", std=" + std + ')';
    }
}
