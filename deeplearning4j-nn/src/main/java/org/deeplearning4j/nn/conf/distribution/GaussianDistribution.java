/*-
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.nn.conf.distribution;

import org.nd4j.shade.jackson.annotation.JsonCreator;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * A normal distribution.
 */

public class GaussianDistribution extends NormalDistribution {

    /**
     * Create a gaussian distribution (equivalent to normal)
     * with the given mean and std
     *
     * @param mean the mean
     * @param std  the standard deviation
     */
    @JsonCreator
    public GaussianDistribution(@JsonProperty("mean") double mean, @JsonProperty("std") double std) {
        super(mean, std);
    }
}
