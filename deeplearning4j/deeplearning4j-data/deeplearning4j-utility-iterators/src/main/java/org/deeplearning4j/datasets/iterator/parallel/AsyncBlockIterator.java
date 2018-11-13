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
        int count = 0;
        for(DataSetIterator iter : newIters){
            log.info("ADDING ITER: " + (count++) + " - hasNext: " + iter.hasNext());
        }
        iteratorsToProcess.addAll(newIters);
    }

    @Override
    public boolean hasAnything() {
        assignIteratorsToThreads();

        //Check iterators, restart any async iterators if required (i.e., if underlying got new data after async shut down)
        for( int i=0; i<asyncIters.length; i++ ){
            softResetIfRequired(i);
        }

        //Check async iterators for next elements
        for(AsyncDataSetIterator iter : asyncIters){
            if(iter != null && iter.hasNext()) {  //May be null: example 2 threads, from 1 source iterator
                return true;
            }
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
            if(asyncIters[i] != null){ //May be null: example 2 threads, from 1 source iterator
                softResetIfRequired(i); //Avoid RC: async iterator might have shut down before more data was added to backing iterator

                if(asyncIters[i].hasNext()) {
                    out[i] = asyncIters[i].next();
                    count++;
                }
            }

            //AsyncDataSetIterator iter = asyncIters[i];
            //log.info( "FIRST LOOP: " + i + " - " + (iter == null ? "NULL" : "Has next: " + iter.hasNext()) + " (backing iter: " + (virtualIters[i] == null ? "NULL" : virtualIters[i].hasNext()) + ")");
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
            for( int j=0; j<asyncIters.length; j++ ){   //TODO let's not always iterate in this order - other async iters could have immediately ready elements
                AsyncDataSetIterator iter = asyncIters[j];
                if(iter == null)    //May be null: example 2 threads, from 1 source iterator
                    continue;
                softResetIfRequired(j); //Avoid RC: async iterator might have shut down before more data was added to backing iterator

                if(iter.hasNext()){
                    out[j] = iter.next();
                    count++;
                }

                anyElementsRemaining |= iter.hasNext();
            }

            if(!anyElementsRemaining){
                //No iters have any elements left
                log.info("NO ITERS HAVE ANY REMAINING");
                int c = 0;
                for(AsyncDataSetIterator iter : asyncIters){
                    log.info( c + " - " + (iter == null ? "NULL" : "Has next: " + iter.hasNext()) + " (backing iter: " + (virtualIters[c] == null ? "NULL" : virtualIters[c].hasNext()) + ")");
                    c++;
                }
                break;
            }
        }

        if(count == maxDataSets ) {
            log.info("RETURNING: " + count + " DATASETS");
            return out;
        }

        //Otherwise, compact array (remove null elements)...
        // TODO do this in a way that keeps device affinity intact as best we can
        org.nd4j.linalg.dataset.api.DataSet[] out2 = new org.nd4j.linalg.dataset.api.DataSet[count];
        int x=0;
        for (org.nd4j.linalg.dataset.api.DataSet ds : out) {
            if (ds == null)
                continue;
            out2[x++] = ds;
        }

        log.info("RETURNING: " + out2.length + " DATASETS (FEWER THAN REQUESTED) - QUEUE SIZE: " + iteratorsToProcess.size());
        return out2;
    }

    protected void softResetIfRequired(int iteratorNum){
        if(asyncIters[iteratorNum] != null && !asyncIters[iteratorNum].hasNext() && asyncIters[iteratorNum].hasNext()){
            log.info("Soft reset of iterator {}", iteratorNum);
            asyncIters[iteratorNum].softReset();
        }
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
                    log.debug("Assigning iterator to device {}", i);
                    Iterator<DataSet> iter = iteratorsToProcess.remove(0);
                    if(iter.hasNext()){
                        virtualIters[i].getIterators().add(iter);
                        if(asyncIters[i] == null){
                            asyncIters[i] = new AsyncDataSetIterator(virtualIters[i], prefetchSize, true, workerThreadDeviceAffinity[i]);
                        }
                    } else {
                        log.warn("Skipping iterator that doesn't have any data");
                        continue;
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

            if(!iteratorsToProcess.isEmpty() && smallestQueueThread >= 0){
                Iterator<DataSet> iter = iteratorsToProcess.remove(0);
                if(iter.hasNext()){
                    virtualIters[smallestQueueThread].getIterators().add(iter);
                    if(asyncIters[smallestQueueThread] == null){
                        asyncIters[smallestQueueThread] = new AsyncDataSetIterator(virtualIters[smallestQueueThread], prefetchSize, true, workerThreadDeviceAffinity[smallestQueueThread]);
                    }
                } else {
                    log.warn("Skipping iterator that doesn't have any data");
                    continue;
                }
                log.debug("Assigning iterator to device {}", smallestQueueThread);
            }
        }
    }
}
