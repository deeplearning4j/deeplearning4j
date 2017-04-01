package org.deeplearning4j.parallelism.inference.observers;

import java.util.Observable;
import java.util.Observer;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.LockSupport;

/**
 * Simple Observer implementation for sequential inference
 *
 * @author raver119@gmail.com
 */
public class BasicInferenceObserver implements Observer {
    private AtomicBoolean finished = new AtomicBoolean(false);

    public BasicInferenceObserver() {
        //
    }

    @Override
    public void update(Observable o, Object arg) {
        finished.set(true);
    }

    /**
     * FOR DEBUGGING ONLY, TO BE REMOVED BEFORE MERGE
     */
    public void waitTillDone(){
        while (!finished.get()) {
            LockSupport.parkNanos(1000);
        }
    }
}
