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

package org.deeplearning4j.spark.data.shuffle;

import lombok.AllArgsConstructor;
import org.apache.spark.Partitioner;

/**
 * A very simple partitioner that assumes integer keys.
 * Maps each value to key % numPartitions
 *
 * @author Alex Black
 * @deprecated Use {@link org.apache.spark.HashPartitioner} instead
 */
@Deprecated
@AllArgsConstructor
public class IntPartitioner extends Partitioner {

    private final int numPartitions;

    @Override
    public int numPartitions() {
        return numPartitions;
    }

    @Override
    public int getPartition(Object key) {
        return (Integer) key % numPartitions;
    }
}
