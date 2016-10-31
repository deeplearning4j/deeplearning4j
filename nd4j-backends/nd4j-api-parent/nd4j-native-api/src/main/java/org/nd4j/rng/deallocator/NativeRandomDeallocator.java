package org.nd4j.rng.deallocator;

import org.nd4j.linalg.api.rng.Random;

import java.lang.ref.ReferenceQueue;

/**
 * Since NativeRandom assumes some native resources, we have to track their use, and deallocate them as soon they are released by JVM GC
 *
 * @author raver119@gmail.com
 */
public class NativeRandomDeallocator {
    private static final NativeRandomDeallocator INSTANCE = new NativeRandomDeallocator();

    private NativeRandomDeallocator() {

    }

    public static NativeRandomDeallocator getInstance() {
        return INSTANCE;
    }




    protected class DeallocatorThread extends Thread implements Runnable {
        private final ReferenceQueue<Random> queue;

        protected DeallocatorThread(int threadId, ReferenceQueue<Random> queue) {
            this.queue = queue;
            this.setName("NativeRandomDeallocator thread " + threadId);
            this.setDaemon(true);
        }

        @Override
        public void run() {
            while (true) {


            }
        }
    }
}
