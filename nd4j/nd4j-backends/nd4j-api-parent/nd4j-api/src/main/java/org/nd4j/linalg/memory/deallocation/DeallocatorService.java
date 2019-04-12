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

package org.nd4j.linalg.memory.deallocation;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.apache.commons.lang3.RandomUtils;
import org.nd4j.linalg.api.memory.Deallocatable;
import org.nd4j.linalg.factory.Nd4j;

import java.lang.ref.ReferenceQueue;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * This class provides unified management for Deallocatable resources
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class DeallocatorService {
    private Thread[] deallocatorThreads;
    private ReferenceQueue<Deallocatable>[] queues;
    private Map<String, DeallocatableReference> referenceMap = new ConcurrentHashMap<>();

    public DeallocatorService() {
        // we need to have at least 2 threads, but for CUDA we'd need at least numDevices threads, due to thread->device affinity
        int numDevices = Nd4j.getAffinityManager().getNumberOfDevices();
        int numThreads = Math.max(2, numDevices * 5);

        deallocatorThreads = new Thread[numThreads];
        queues = new ReferenceQueue[numThreads];
        for (int e = 0; e < numThreads; e++) {
            log.debug("Starting deallocator thread {}", e + 1);
            queues[e] = new ReferenceQueue<>();

            // attaching queue to its own thread
            deallocatorThreads[e] = new DeallocatorServiceThread(queues[e]);
            deallocatorThreads[e].setName("DeallocatorServiceThread_" + e);
            deallocatorThreads[e].setDaemon(true);

            // optionally setting up affinity
            //if (numDevices > 1)
            //    Nd4j.getAffinityManager().attachThreadToDevice(deallocatorThreads[e], e);

            deallocatorThreads[e].start();
        }
    }

    /**
     * This method adds Deallocatable object instance to tracking system
     *
     * @param deallocatable object to track
     * @param bucketId ID of the bucked. In multi-device systems each object is tied to own device, with 1:1 mapping
     */
    public void pickObject(@NonNull Deallocatable deallocatable, int bucketId) {
        val reference = new DeallocatableReference(deallocatable, queues[bucketId]);
        referenceMap.put(deallocatable.getUniqueId(), reference);
    }

    /**
     * This method adds Deallocatable object instance to tracking system
     *
     * @param deallocatable object to track
     */
    public void pickObject(@NonNull Deallocatable deallocatable) {
        // using rng here, to spread load among buckets
        pickObject(deallocatable, RandomUtils.nextInt(0, queues.length));
    }


    private class DeallocatorServiceThread extends Thread implements Runnable {
        private ReferenceQueue<Deallocatable> queue;

        private DeallocatorServiceThread(@NonNull ReferenceQueue<Deallocatable> queue) {
            this.queue = queue;
        }

        @Override
        public void run() {
            boolean canRun = true;
            long cnt = 0;
            while (canRun) {
                try {
                    val reference = (DeallocatableReference) queue.remove();
                    if (reference == null)
                        continue;

                    // invoking deallocator
                    reference.getDeallocator().deallocate();
                    referenceMap.remove(reference.getId());
                } catch (InterruptedException e) {
                    canRun = false;
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            }
        }
    }
}
