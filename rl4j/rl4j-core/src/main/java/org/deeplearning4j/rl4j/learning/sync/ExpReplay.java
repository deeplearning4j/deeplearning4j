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

package org.deeplearning4j.rl4j.learning.sync;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.queue.CircularFifoQueue;

import java.util.*;
import java.util.concurrent.ThreadLocalRandom;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) 7/12/16.
 *
 * "Standard" Exp Replay implementation that uses a CircularFifoQueue
 *
 * The memory is optimised by using array of INDArray in the transitions
 * such that two same INDArrays are not allocated twice
 */
@Slf4j
public class ExpReplay<A> implements IExpReplay<A> {

    final private int batchSize;
    final private Random random;

    //Implementing this as a circular buffer queue
    private CircularFifoQueue<Transition<A>> storage;

    public ExpReplay(int maxSize, int batchSize, int seed) {
        this.batchSize = batchSize;
        this.random = new Random(seed);
        storage = new CircularFifoQueue<>(maxSize);
    }


    public ArrayList<Transition<A>> getBatch(int size) {
        ArrayList<Transition<A>> batch = new ArrayList<>(size);
        int storageSize = storage.size();
        int actualBatchSize = Math.min(storageSize, size);
        int[] actualIndex = ThreadLocalRandom.current().ints(0, storageSize).distinct().limit(actualBatchSize).toArray();

        for (int i = 0; i < actualBatchSize; i ++) {
            Transition<A> trans = storage.get(actualIndex[i]);
            batch.add(trans.dup());
        }

        return batch;
    }

    public ArrayList<Transition<A>> getBatch() {
        return getBatch(batchSize);
    }

    public void store(Transition<A> transition) {
        storage.add(transition);
        //log.info("size: "+storage.size());
    }



}
