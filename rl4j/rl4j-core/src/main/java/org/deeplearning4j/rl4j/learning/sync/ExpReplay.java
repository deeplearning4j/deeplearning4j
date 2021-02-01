/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.rl4j.learning.sync;

import it.unimi.dsi.fastutil.ints.IntOpenHashSet;
import it.unimi.dsi.fastutil.ints.IntSet;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.queue.CircularFifoQueue;
import org.deeplearning4j.rl4j.experience.StateActionRewardState;
import org.nd4j.linalg.api.rng.Random;

import java.util.ArrayList;

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
    final private Random rnd;

    //Implementing this as a circular buffer queue
    private CircularFifoQueue<StateActionRewardState<A>> storage;

    public ExpReplay(int maxSize, int batchSize, Random rnd) {
        this.batchSize = batchSize;
        this.rnd = rnd;
        storage = new CircularFifoQueue<>(maxSize);
    }

    public ArrayList<StateActionRewardState<A>> getBatch(int size) {
        ArrayList<StateActionRewardState<A>> batch = new ArrayList<>(size);
        int storageSize = storage.size();
        int actualBatchSize = Math.min(storageSize, size);

        int[] actualIndex = new int[actualBatchSize];
        IntSet set = new IntOpenHashSet();
        for( int i=0; i<actualBatchSize; i++ ){
            int next = rnd.nextInt(storageSize);
            while(set.contains(next)){
                next = rnd.nextInt(storageSize);
            }
            set.add(next);
            actualIndex[i] = next;
        }

        for (int i = 0; i < actualBatchSize; i ++) {
            StateActionRewardState<A> trans = storage.get(actualIndex[i]);
            batch.add(trans.dup());
        }

        return batch;
    }

    public ArrayList<StateActionRewardState<A>> getBatch() {
        return getBatch(batchSize);
    }

    public void store(StateActionRewardState<A> stateActionRewardState) {
        storage.add(stateActionRewardState);
        //log.info("size: "+storage.size());
    }

    @Override
    public int getDesignatedBatchSize() {
        return batchSize;
    }

    public int getBatchSize() {
        int storageSize = storage.size();
        return Math.min(storageSize, batchSize);
    }

}
