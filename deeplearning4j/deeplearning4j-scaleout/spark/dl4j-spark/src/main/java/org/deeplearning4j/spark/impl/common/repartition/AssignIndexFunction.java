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

import org.apache.spark.api.java.function.Function2;
import org.apache.spark.rdd.RDD;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * A Function2 used to assign each element in a RDD an index (integer key). This is used later in the {@link BalancedPartitioner}
 * to enable partitioning to be done in a way that is more reliable (less random) than standard .repartition calls
 *
 * @author Alex Black
 * @deprecated Use {@link RDD#zipWithIndex()} instead
 */
@Deprecated
public class AssignIndexFunction<T> implements Function2<Integer, Iterator<T>, Iterator<Tuple2<Integer, T>>> {
    private final int[] partitionElementStartIdxs;

    /**
     * @param partitionElementStartIdxs    These are the start indexes for elements in each partition (determined from the
     *                                     number of elements in each partition). Thus length of the array must be equal
     *                                     to the number of partitions
     */
    public AssignIndexFunction(int[] partitionElementStartIdxs) {
        this.partitionElementStartIdxs = partitionElementStartIdxs;
    }

    @Override
    public Iterator<Tuple2<Integer, T>> call(Integer partitionNum, Iterator<T> v2) throws Exception {
        int currIdx = partitionElementStartIdxs[partitionNum];
        List<Tuple2<Integer, T>> list = new ArrayList<>();
        while (v2.hasNext()) {
            list.add(new Tuple2<>(currIdx++, v2.next()));
        }
        return list.iterator();
    }
}
