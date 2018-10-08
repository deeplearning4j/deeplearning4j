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
 * A log-normal distribution, with two parameters: mean and standard deviation.
 * Note: the mean and standard deviation are for the logarithm of the values.
 * Put another way: if X~LogN(M,S), then mean(log(X))=M, and stdev(log(X))=S
 *
 */
@EqualsAndHashCode(callSuper = false)
@Data
public class LogNormalDistribution extends Distribution {

    private double mean, std;

    /**
     * Create a log-normal distribution
     * with the given mean and std
     *
     * @param mean the mean
     * @param std  the standard deviation
     */
    @JsonCreator
    public LogNormalDistribution(@JsonProperty("mean") double mean, @JsonProperty("std") double std) {
        this.mean = mean;
        this.std = std;
    }

    public String toString() {
        return "LogNormalDistribution(" + "mean=" + mean + ", std=" + std + ')';
    }
}
