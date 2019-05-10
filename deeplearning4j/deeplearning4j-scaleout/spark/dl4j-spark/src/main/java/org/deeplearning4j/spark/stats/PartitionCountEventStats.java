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

package org.deeplearning4j.spark.stats;

import lombok.Getter;

/**
 * Event stats implementation with partition count
 *
 * @author Alex Black
 */
public class PartitionCountEventStats extends BaseEventStats {

    @Getter
    private final int numPartitions;

    public PartitionCountEventStats(long startTime, long durationMs, int numPartitions) {
        super(startTime, durationMs);
        this.numPartitions = numPartitions;
    }

    public PartitionCountEventStats(String machineId, String jvmId, long threadId, long startTime, long durationMs,
                    int numPartitions) {
        super(machineId, jvmId, threadId, startTime, durationMs);
        this.numPartitions = numPartitions;
    }

    @Override
    public String asString(String delimiter) {
        return super.asString(delimiter) + delimiter + numPartitions;
    }

    @Override
    public String getStringHeader(String delimiter) {
        return super.getStringHeader(delimiter) + delimiter + "numPartitions";
    }
}
