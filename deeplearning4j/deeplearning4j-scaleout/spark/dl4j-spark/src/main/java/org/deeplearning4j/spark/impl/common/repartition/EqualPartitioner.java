/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.spark.impl.common.repartition;

import lombok.AllArgsConstructor;
import lombok.EqualsAndHashCode;
import lombok.extern.slf4j.Slf4j;
import org.apache.spark.Partitioner;
import org.apache.spark.api.java.JavaRDD;

import java.util.Random;

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
