package org.nd4j.linalg.heartbeat;

import org.nd4j.linalg.heartbeat.reports.Environment;
import org.nd4j.linalg.heartbeat.reports.Event;
import org.nd4j.linalg.heartbeat.reports.Task;

import java.util.concurrent.atomic.AtomicLong;

/**
 *
 * Heartbeat implementation for ND4j
 *
 * @author raver119@gmail.com
 */
public class Heartbeat {
    private static final Heartbeat INSTANCE = new Heartbeat();
    private AtomicLong lastReport = new AtomicLong(0);
    private long serialVersionID;


    protected Heartbeat() {

    }

    public static Heartbeat getInstance() {
        return INSTANCE;
    }

    public synchronized void reportEvent(Event event, Environment environment, Task task) {
        // TODO: to be implemented
        /*
            Basically we need to start background thread here, but only if lastReport was long ago, to avoid spam.
         */
        long currentTime = System.currentTimeMillis();
        // set to one-time message for now
        if (lastReport.get() == 0) {
            lastReport.set(currentTime);

            RepoThread thread = new RepoThread(environment, null);
            thread.start();
        }
    }

    public synchronized void derivedId(long id) {
        /*
            this method will be actually used only after ast is implemented
         */
        if (serialVersionID != 0) {
            serialVersionID = id;
        }
    }

    private class RepoThread extends Thread implements Runnable {
        /**
         * Thread for quiet background reporting.
         */
        private final Environment environment;
        private final Task task;


        public RepoThread(Environment environment, Task task) {
            this.environment = environment;
            this.task = task;
        }

        @Override
        public void run() {
            try {

            } catch (Exception e) {
                // keep quiet on exceptions here
                ;
            } finally {
                ;
            }
        }
    }
}
