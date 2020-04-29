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

package org.nd4j.linalg.api.ops.aggregates;

import org.nd4j.shade.guava.collect.Lists;
import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.exception.ND4JIllegalStateException;

import java.util.ArrayList;
import java.util.List;

/**
 * Wrapper for "batch of aggregates"
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class Batch<T extends Aggregate> {
    /**
     * This batchLimit should be equal to its counterpart at helper_ptrmap.h
     *
     */
    @Getter
    @Setter
    private DataBuffer paramsSurface;

    @Getter
    private static final int batchLimit = 512;

    // all aggregates within this batch
    @Getter
    private List<T> aggregates;

    @Getter
    private T sample;
    @Getter
    private int numAggregates;

    /**
     * This constructor takes List of Aggregates, and builds Batch instance, usable with Nd4j executioner.
     *
     * @param aggregates
     */
    public Batch(List<T> aggregates) {
        //if (aggregates.size() > batchLimit)
        //    throw new RuntimeException("Number of aggregates is higher then " + batchLimit + " elements, multiple batches should be issued.");

        this.aggregates = aggregates;
        this.numAggregates = aggregates.size();

        // we fetch single sample for possible future use. not sure if will be used though
        this.sample = aggregates.get(0);
    }

    /**
     * This method returns opNum for batched aggregate
     * @return
     */
    public int opNum() {
        return sample.opNum();
    }

    /**
     * This method tries to append aggregate to the current batch, if it has free room
     *
     * @param aggregate
     * @return
     */
    public boolean append(T aggregate) {
        if (!isFull()) {
            aggregates.add(aggregate);
            return true;
        } else
            return false;
    }

    /**
     * This method checks, if number of batched aggregates equals to maximum possible value
     *
     * @return
     */
    public boolean isFull() {
        return batchLimit == numAggregates;
    }


    /**
     * Helper method to create batch from list of aggregates, for cases when list of aggregates is higher then batchLimit
     *
     * @param list
     * @param <U>
     * @return
     */
    public static <U extends Aggregate> List<Batch<U>> getBatches(List<U> list) {
        return getBatches(list, batchLimit);
    }

    /**
     * Helper method to create batch from list of aggregates, for cases when list of aggregates is higher then batchLimit
     *
     * @param list
     * @param <U>
     * @return
     */
    public static <U extends Aggregate> List<Batch<U>> getBatches(List<U> list, int partitionSize) {
        DataType c = null;
        for (val u:list) {
            for (val a:u.getArguments()) {
                // we'll be comparing to the first array
                if (c == null && a != null)
                    c = a.dataType();

                if (a != null && c != null)
                    Preconditions.checkArgument(c == a.dataType(), "All arguments must have same data type");
            }
        }

        if (c == null)
            throw new ND4JIllegalStateException("Can't infer data type from arguments");

        List<List<U>> partitions = Lists.partition(list, partitionSize);
        List<Batch<U>> split = new ArrayList<>();

        for (List<U> partition : partitions) {
            split.add(new Batch<U>(partition));
        }

        return split;
    }
}
