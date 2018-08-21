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

package org.deeplearning4j.spark.impl.common.repartition;

import lombok.AllArgsConstructor;
import lombok.EqualsAndHashCode;
import lombok.extern.slf4j.Slf4j;
import org.apache.spark.Partitioner;
import org.apache.spark.api.java.JavaRDD;

import java.util.Random;

/**
 * This is a custom partitioner (used in conjunction with {@link JavaRDD#zipWithIndex()} to repartition a RDD.
 * Unlike a standard .repartition() call (which assigns partitions like [2,3,4,1,2,3,4,1,2,...] for 4 partitions],
 * this function attempts to keep contiguous elements (i.e., those elements originally in the same partition) together
 * much more frequently. Furthermore, it is less prone to producing larger or smaller than expected partitions, as
 * it is entirely deterministic, whereas .repartition() has a degree of randomness (i.e., start index) which can result in
 * a large degree of variance when the number of elements in the original partitions is small (as is the case generally in DL4J)<br>
 * Note also that if the number of elements are not a multiple of the number of partitions, an int[] to specify the
 * locations of these values is used instead.
 *
 * @author Alex Black
 */
@Slf4j
@AllArgsConstructor
public class EqualPartitioner extends Partitioner {
    private final int numPartitions; //Total number of partitions
    private final int partitionSizeExRemainder;
    private final int[] remainderPositions;

    @Override
    public int numPartitions() {
        return numPartitions;
    }

    @Override
    public int getPartition(Object key) {
        int elementIdx = key.hashCode();

        //Assign an equal number of elements to each partition, sequentially
        // For any remainder, use the specified remainder indexes

        //Work out: which partition it belongs to...
        if(elementIdx < numPartitions * partitionSizeExRemainder){
            //Standard element
            return elementIdx / partitionSizeExRemainder;
        } else {
            //Is a 'remainder' element
            int remainderNum = elementIdx % numPartitions;
            return remainderPositions[remainderNum %remainderPositions.length];     //Final mod here shouldn't be necessary, but is here for safety...
        }
    }
}
