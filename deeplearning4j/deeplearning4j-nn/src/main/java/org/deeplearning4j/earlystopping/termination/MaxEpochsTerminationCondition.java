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
