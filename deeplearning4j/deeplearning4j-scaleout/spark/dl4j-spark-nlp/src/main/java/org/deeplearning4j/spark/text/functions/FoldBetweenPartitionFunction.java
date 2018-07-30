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

package org.deeplearning4j.spark.text.functions;

import org.apache.spark.api.java.function.Function2;
import org.apache.spark.broadcast.Broadcast;
import org.nd4j.linalg.primitives.Counter;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

/**
 * @author jeffreytang
 */
public class FoldBetweenPartitionFunction implements Function2<Integer, Iterator<AtomicLong>, Iterator<Long>> {
    private Broadcast<Counter<Integer>> broadcastedMaxPerPartitionCounter;

    public FoldBetweenPartitionFunction(Broadcast<Counter<Integer>> broadcastedMaxPerPartitionCounter) {
        this.broadcastedMaxPerPartitionCounter = broadcastedMaxPerPartitionCounter;
    }

    @Override
    public Iterator<Long> call(Integer ind, Iterator<AtomicLong> partition) throws Exception {
        int sumToAdd = 0;
        Counter<Integer> maxPerPartitionCounterInScope = broadcastedMaxPerPartitionCounter.value();

        // Add the sum of counts of all the partition with an index lower than the current one
        if (ind != 0) {
            for (int i = 0; i < ind; i++) {
                sumToAdd += maxPerPartitionCounterInScope.getCount(i);
            }
        }

        // Add the sum of counts to each element of the partition
        List<Long> itemsAddedToList = new ArrayList<>();
        while (partition.hasNext()) {
            itemsAddedToList.add(partition.next().get() + sumToAdd);
        }

        return itemsAddedToList.iterator();
    }
}
