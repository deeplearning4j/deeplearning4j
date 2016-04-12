package org.nd4j.linalg.heartbeat;


import org.nd4j.linalg.heartbeat.reports.Environment;
import org.nd4j.linalg.heartbeat.reports.Event;
import org.nd4j.linalg.heartbeat.reports.Task;

import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 *
 * Heartbeat implementation for ND4j
 *
 * @author raver119@gmail.com
 */
public class Heartbeat {
    private static final Heartbeat INSTANCE = new Heartbeat();
    private volatile long serialVersionID;
    private AtomicBoolean enabled = new AtomicBoolean(true);


    protected Heartbeat() {

    }

    public static Heartbeat getInstance() {
        return INSTANCE;
    }

    public void disableHeartbeat() {
        this.enabled.set(false);
    }

    public synchronized void reportEvent(Event event, Environment environment, Task task) {

    }

    public synchronized void derivedId(long id) {

    }

    private synchronized long getDerivedId() {
        return serialVersionID;
    }

    private class RepoThread extends Thread implements Runnable {
        /**
         * Thread for quiet background reporting.
         */
        private final Environment environment;
        private final Task task;
        private final Event event;


        public RepoThread(Event event, Environment environment, Task task) {
            this.environment = environment;
            this.task = task;
            this.event = event;
        }

        @Override
        public void run() {

        }
    }

}
