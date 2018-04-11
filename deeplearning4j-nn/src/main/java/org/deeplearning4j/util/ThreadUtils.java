package org.deeplearning4j.util;

import java.util.concurrent.locks.LockSupport;

/**
 * Utils for the basic use and flow of threads.
 */
public class ThreadUtils {
    public static void uncheckedSleep(long millis) {
        LockSupport.parkNanos(millis * 1000000);
        // we must check the interrupted status in case this is used in a loop
        // Otherwise we may end up spinning 100% without breaking out on an interruption
        if (Thread.currentThread().isInterrupted()) {
            throw new UncheckedInterruptedException();
        }
    }
    
    /**
     * Similar to {@link InterruptedException} in concept, but unchecked.  Allowing this to be thrown without being 
     * explicitly declared in the API.
     */
    public static class UncheckedInterruptedException extends RuntimeException {
	
    }
}
