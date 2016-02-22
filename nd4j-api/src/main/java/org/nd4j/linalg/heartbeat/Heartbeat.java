package org.nd4j.linalg.heartbeat;

import com.dmurph.tracking.AnalyticsConfigData;
import com.dmurph.tracking.JGoogleAnalyticsTracker;
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
    private AtomicLong lastReport = new AtomicLong(0);
    private AtomicBoolean enabled = new AtomicBoolean(true);
    private long cId;
    private volatile JGoogleAnalyticsTracker tracker;


    protected Heartbeat() {
        try {
            AnalyticsConfigData config = new AnalyticsConfigData("MyTrackingCode");
            tracker = new JGoogleAnalyticsTracker(config, JGoogleAnalyticsTracker.GoogleAnalyticsVersion.V_4_7_2);
            tracker.setEnabled(true);
        } catch (Exception e) {
            ; // do nothing here
        }
    }

    public static Heartbeat getInstance() {
        return INSTANCE;
    }

    public void disableHeartbeat() {
        this.enabled.set(false);
    }

    public synchronized void reportEvent(Event event, Environment environment, Task task) {
        // Ignore reported even if heartbeat is disabled
        if (!enabled.get()) return;
        /*
            Basically we need to start background thread here, but only if lastReport was long ago, to avoid spam.
         */
        long currentTime = System.currentTimeMillis();
        // set to one-time message for now
        if (lastReport.get() == 0) {
            lastReport.set(currentTime);

            RepoThread thread = new RepoThread(environment, task);
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

    private synchronized long getDerivedId() {
        return serialVersionID;
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
                //tracker.trackEvent();
            } catch (Exception e) {
                // keep quiet on exceptions here
                ;
            } finally {
                ;
            }
        }
    }
}
