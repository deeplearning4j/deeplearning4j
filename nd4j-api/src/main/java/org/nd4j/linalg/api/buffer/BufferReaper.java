/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 *
 */

package org.nd4j.linalg.api.buffer;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.lang.ref.Reference;
import java.lang.ref.ReferenceQueue;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Buffer reaper for handling freeing of resources
 * related to buffers. NDArrays themselves are lightweight objects
 * that tend to be referenced for GC early, but the buffers often stick around.
 * The solution is as follows:
 * Given an ndarrays's id, pass the id
 * as a referencing item to the buffer.
 * <p/>
 * The buffer knows what items are referencing it
 * without directly holding a reference to the object itself.
 * <p/>
 * The major reason we need to do this is because of how reference counting works.
 * <p/>
 * When we do reference counting, we don't want the garbage collector to be
 * tricked in to thinking that the object is still being used when in reality its a circular reference.
 * This id mechanism allows tracking while still allowing the GC to properly mark items
 * for collection.
 *
 * @author Adam Gibson
 */
public class BufferReaper extends Thread {
    private ReferenceQueue<INDArray> queue;
    private ReferenceQueue<DataBuffer> buffer;
    private AtomicLong ranFinals;

    public BufferReaper(ReferenceQueue<INDArray> queue, ReferenceQueue<DataBuffer> buffer) {
        this.queue = queue;
        this.buffer = buffer;
        init(queue,buffer);

    }

    public BufferReaper(Runnable target, ReferenceQueue<INDArray> queue, ReferenceQueue<DataBuffer> buffer) {
        super(target);
        this.queue = queue;
        this.buffer = buffer;
        init(queue,buffer);

    }

    public BufferReaper(ThreadGroup group, Runnable target, ReferenceQueue<INDArray> queue, ReferenceQueue<DataBuffer> buffer) {
        super(group, target);
        this.queue = queue;
        this.buffer = buffer;
        init(queue,buffer);

    }

    public BufferReaper(String name, ReferenceQueue<INDArray> queue, ReferenceQueue<DataBuffer> buffer) {
        super(name);
        this.queue = queue;
        this.buffer = buffer;
        init(queue,buffer);

    }

    public BufferReaper(ThreadGroup group, String name, ReferenceQueue<INDArray> queue, ReferenceQueue<DataBuffer> buffer) {
        super(group, name);
        this.queue = queue;
        this.buffer = buffer;
        init(queue,buffer);

    }

    public BufferReaper(Runnable target, String name, ReferenceQueue<INDArray> queue, ReferenceQueue<DataBuffer> buffer) {
        super(target, name);
        this.queue = queue;
        this.buffer = buffer;
        init(queue,buffer);

    }

    public BufferReaper(ThreadGroup group, Runnable target, String name, ReferenceQueue<INDArray> queue, ReferenceQueue<DataBuffer> buffer) {
        super(group, target, name);
        this.queue = queue;
        this.buffer = buffer;
        init(queue,buffer);

    }

    public BufferReaper(ThreadGroup group, Runnable target, String name, long stackSize, ReferenceQueue<INDArray> queue, ReferenceQueue<DataBuffer> buffer) {
        super(group, target, name, stackSize);
        this.queue = queue;
        this.buffer = buffer;
        init(queue,buffer);
    }

    private void init(ReferenceQueue<INDArray> queue,ReferenceQueue<DataBuffer> buffer) {
        this.queue = queue;
        this.buffer =  buffer;
        setName("BufferCleanup");
        setDaemon(true);
        ranFinals = new AtomicLong(-1);
    }

    /**
     * Frees data used by the given ndarrays
     * @param arrs the arrays to free
     */
    public static  void destroy(INDArray...arrs) {
//        for(INDArray arr : arrs)
//            arr.data().destroy();
    }




    @Override
    public void run() {

        while (Nd4j.resourceManagerOn) {
            Reference<INDArray> queue2 = (Reference<INDArray>) queue.poll();
            while(queue2 != null) {
                queue2.get().cleanup();
                queue2 = (Reference<INDArray>) queue.poll();
            }

            Reference<DataBuffer> bufferQueue = (Reference<DataBuffer>) buffer.poll();
            while(bufferQueue != null) {
                //bufferQueue.get().destroy();
                bufferQueue = (Reference<DataBuffer>) buffer.poll();
            }

            //Nd4j.getResourceManager().purge();
            
            try {
				Thread.sleep(1000);
			} catch (InterruptedException e) {
			}
        }
    }
}
