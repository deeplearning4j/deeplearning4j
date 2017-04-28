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

package org.deeplearning4j.earlystopping.termination;

import lombok.NoArgsConstructor;
import org.nd4j.shade.jackson.annotation.JsonCreator;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/** Terminate training if the number of epochs exceeds the maximum number of epochs */
@NoArgsConstructor
public class MaxEpochsTerminationCondition implements EpochTerminationCondition {
    @JsonProperty
    private int maxEpochs;

    @JsonCreator
    public MaxEpochsTerminationCondition(int maxEpochs) {
        if (maxEpochs <= 0)
            throw new IllegalArgumentException("Max number of epochs must be >= 1");
        this.maxEpochs = maxEpochs;
    }

    @Override
    public void initialize() {
        //No op
    }

    @Override
    public boolean terminate(int epochNum, double score) {
        return epochNum + 1 >= maxEpochs; //epochNum starts at 0
    }

    @Override
    public String toString() {
        return "MaxEpochsTerminationCondition(" + maxEpochs + ")";
    }
}
