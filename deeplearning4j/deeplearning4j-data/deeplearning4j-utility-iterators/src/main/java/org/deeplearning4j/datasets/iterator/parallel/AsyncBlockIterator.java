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

package org.deeplearning4j.datasets.iterator.parallel;

import lombok.Getter;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.BlockDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.*;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.CopyOnWriteArrayList;

@Slf4j
public class AsyncBlockIterator implements BlockDataSetIterator {

    private final int[] workerThreadDeviceAffinity;
    private final int prefetchSize;
    private final List<Iterator<DataSet>> iteratorsToProcess;

    //Array index: thread number
    @Getter
    private final AsyncDataSetIterator[] asyncIters;
    @Getter
    private final VirtualDataSetIterator[] virtualIters;

    public AsyncBlockIterator(@NonNull int[] workerThreadDeviceAffinity, int prefetchSize, List<Iterator<DataSet>> backingList ){
        Preconditions.checkState(workerThreadDeviceAffinity.length > 0, "Number of devices (workerThreadDeviceAffinity.length) must be > 0 - got %s", workerThreadDeviceAffinity.length);
        this.workerThreadDeviceAffinity = workerThreadDeviceAffinity;
        this.prefetchSize = prefetchSize;
        if(backingList == null){
            this.iteratorsToProcess = new CopyOnWriteArrayList<>();
        } else {
            this.iteratorsToProcess = backingList;
        }


        asyncIters = new AsyncDataSetIterator[workerThreadDeviceAffinity.length];
        virtualIters = new VirtualDataSetIterator[workerThreadDeviceAffinity.length];
        for( int i=0; i<workerThreadDeviceAffinity.length; i++ ){
            virtualIters[i] = new VirtualDataSetIterator(new ArrayList<Iterator<DataSet>>());
        }
    }

    public void attach(@NonNull Collection<DataSetIterator> newIters){
        iteratorsToProcess.addAll(newIters);
    }

    @Override
    public boolean hasAnything() {
        boolean any = iteratorsToProcess.size() > 0;
        assignIteratorsToThreads();

        //Check iterators, restart any async iterators if required
        for( int i=0; i<asyncIters.length; i++ ){
            if(virtualIters[i].hasNext()){
                if(!asyncIters[i].hasNext()){
                    //Async iterator probably finished just before virtual iterator had more data added
                    asyncIters[i].softReset();
                }
            }
        }

        if(any)
            return true;

        //Check async iterators for next elements
        for(AsyncDataSetIterator iter : asyncIters){
            if(iter.hasNext())
                return true;
        }

        return false;
    }

    @Override
    public org.nd4j.linalg.dataset.api.DataSet[] next(int maxDataSets) {
        if(!hasAnything())
            throw new NoSuchElementException("No remaining elements");
        Preconditions.checkState(maxDataSets > 0 && maxDataSets <= workerThreadDeviceAffinity.length, "Max data sets must be in" +
                " range 1 to %s inclusive, got %s", workerThreadDeviceAffinity.length, maxDataSets);

        //Try to maintain existing thread-device affinity by fetching DataSets from corresponding iterators
        org.nd4j.linalg.dataset.api.DataSet[] out = new org.nd4j.linalg.dataset.api.DataSet[maxDataSets];
        int count = 0;
        for( int i=0; i<maxDataSets; i++ ){
            if(asyncIters[i].hasNext()){
                out[i] = asyncIters[i].next();
                count++;
            }
        }

        if(count == maxDataSets)
            return out;

        //Otherwise: one or more async prefetch threads don't have any next elements. Let's try to get from another source,
        // even if we have to relocate DataSets between devices to do so

        for( int i=0; i<maxDataSets; i++ ) {
            if(out[i] != null){
                continue;
            }

            boolean anyElementsRemaining = false;
            for(AsyncDataSetIterator iter : asyncIters){        //TODO let's not always iterate in this order - other async iters could have ready elements
                if(iter.hasNext()){
                    out[i] = iter.next();
                    count++;
                }

                anyElementsRemaining |= iter.hasNext();
            }

            if(!anyElementsRemaining){
                //No iters have any elements left
                break;
            }
        }

        if(count == maxDataSets )
            return out;

        //Otherwise, compact array (remove null elements)...
        // TODO do this in a way that keeps device affinity intact as best we can
        org.nd4j.linalg.dataset.api.DataSet[] out2 = new org.nd4j.linalg.dataset.api.DataSet[count];
        int x=0;
        for (org.nd4j.linalg.dataset.api.DataSet ds : out) {
            if (ds == null)
                continue;
            out2[x++] = ds;
        }
        return out2;
    }



    protected synchronized void assignIteratorsToThreads(){
        if(iteratorsToProcess.isEmpty())
            return;

        //Challenge 1: Workers may be feeding at different rates
        //Challenge 2: DataSetIterators may have different number of examples in each

        //Assignment algorithm: assign to the thread with the smallest queue (or empty queue)
        while(!iteratorsToProcess.isEmpty()) {
            int smallestQueueSize = Integer.MAX_VALUE;
            int smallestQueueThread = -1;
            for (int i = 0; i < workerThreadDeviceAffinity.length; i++) {
                if (!virtualIters[i].hasNext()) {
                    //Empty
                    log.info("Assigning iterator to device {}", i);
                    virtualIters[i].getIterators().add((Iterator<DataSet>) iteratorsToProcess.remove(0));
                    if(asyncIters[i] == null){
                        asyncIters[i] = new AsyncDataSetIterator(virtualIters[i], prefetchSize, true, workerThreadDeviceAffinity[i]);
                    }
                    break;
                } else {
                    int currQueueSize = virtualIters[i].getIterators().size() - virtualIters[i].getPosition().get();
                    if(currQueueSize < smallestQueueSize){
                        smallestQueueSize = currQueueSize;
                        smallestQueueThread = i;
                    }
                }
            }

            if(smallestQueueThread >= 0){
                virtualIters[smallestQueueThread].getIterators().add((Iterator<DataSet>) iteratorsToProcess.remove(0));
                if(asyncIters[smallestQueueThread] == null){
                    asyncIters[smallestQueueThread] = new AsyncDataSetIterator(virtualIters[smallestQueueThread], prefetchSize, true, workerThreadDeviceAffinity[smallestQueueThread]);
                }
                log.info("Assigning iterator to device {}", smallestQueueThread);
            }
        }
    }
}
