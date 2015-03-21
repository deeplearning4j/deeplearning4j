package org.nd4j.linalg.api.buffer;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.lang.ref.Reference;
import java.lang.ref.ReferenceQueue;
import java.util.concurrent.TimeUnit;
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
    private AtomicLong ranFinals;

    public BufferReaper(ReferenceQueue<INDArray> queue) {
        init(queue);
    }

    public BufferReaper(Runnable target, ReferenceQueue<INDArray> queue) {
        super(target);
        init(queue);
    }

    public BufferReaper(ThreadGroup group, Runnable target, ReferenceQueue<INDArray> queue) {
        super(group, target);
        init(queue);
    }

    public BufferReaper(String name, ReferenceQueue<INDArray> queue) {
        super(name);
        init(queue);
    }

    public BufferReaper(ThreadGroup group, String name, ReferenceQueue<INDArray> queue) {
        super(group, name);
        init(queue);
    }

    public BufferReaper(Runnable target, String name, ReferenceQueue<INDArray> queue) {
        super(target, name);
        init(queue);
    }

    public BufferReaper(ThreadGroup group, Runnable target, String name, ReferenceQueue<INDArray> queue) {
        super(group, target, name);
        init(queue);
    }

    public BufferReaper(ThreadGroup group, Runnable target, String name, long stackSize, ReferenceQueue<INDArray> queue) {
        super(group, target, name, stackSize);
        init(queue);
    }

    private void init(ReferenceQueue<INDArray> queue) {
        this.queue = queue;
        setPriority(Thread.MAX_PRIORITY);
        setName("BufferCleanup");
        setDaemon(true);
        ranFinals = new AtomicLong(-1);
    }


    private void runFinalize() {
        long curr = System.currentTimeMillis();
        long old = ranFinals.get();
        if (old < 0)
            ranFinals.set(curr);
        else {
            long delta = Math.abs(curr - old);
            long seconds = TimeUnit.MILLISECONDS.toSeconds(delta);
            if (seconds >= 60) {
                System.gc();
                System.runFinalization();
                ranFinals.set(System.currentTimeMillis());
            }
        }
    }

    @Override
    public void run() {

        while (true) {
            Reference<INDArray> ref = (Reference<INDArray>) queue.poll();
            runFinalize();
            if (ref != null) {
                INDArray reffed = ref.get();
                //remove the reference since this will be gced
                reffed.data().removeReferencing(reffed.id());

            }

        }
    }
}
