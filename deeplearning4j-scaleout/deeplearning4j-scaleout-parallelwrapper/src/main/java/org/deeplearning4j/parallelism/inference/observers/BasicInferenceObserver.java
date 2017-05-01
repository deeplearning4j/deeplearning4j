package org.deeplearning4j.parallelism.inference.observers;

import lombok.extern.slf4j.Slf4j;

import java.util.Observable;
import java.util.Observer;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.LockSupport;

/**
 * Simple Observer implementation for
 * sequential inference
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class BasicInferenceObserver implements Observer {
    private AtomicBoolean finished;

    public BasicInferenceObserver() {
        finished = new AtomicBoolean(false);
    }

    @Override
    public void update(Observable o, Object arg) {
        finished.set(true);
    }

    /**
     * FOR DEBUGGING ONLY, TO BE REMOVED BEFORE MERGE
     */
    public void waitTillDone() {
        while (!finished.get()) {
            LockSupport.parkNanos(1000);
        }
    }
}
