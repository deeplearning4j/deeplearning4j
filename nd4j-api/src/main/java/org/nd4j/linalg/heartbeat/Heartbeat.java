package org.nd4j.linalg.heartbeat;

import com.brsanthu.googleanalytics.GoogleAnalytics;
import com.brsanthu.googleanalytics.GoogleAnalyticsRequest;
import com.brsanthu.googleanalytics.PageViewHit;
import org.nd4j.linalg.heartbeat.reports.Environment;
import org.nd4j.linalg.heartbeat.reports.Event;
import org.nd4j.linalg.heartbeat.reports.Task;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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
    private volatile GoogleAnalytics tracker;
    private static Logger logger = LoggerFactory.getLogger(Heartbeat.class);


    protected Heartbeat() {
        try {
            tracker = new GoogleAnalytics("UA-48811288-4");
        } catch (Exception e) {
            ; // do nothing here
            throw new RuntimeException(e);
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

            RepoThread thread = new RepoThread(event, environment, task);
            thread.start();
        } else System.out.println("Skipping report");
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
        private final Event event;


        public RepoThread(Event event, Environment environment, Task task) {
            this.environment = environment;
            this.task = task;
            this.event = event;
        }

        @Override
        public void run() {
            try {
                /*
                 now we just should pack everything environment/task into single line
                 */
                String lid = this.event.toString();
                String argAction = serialVersionID != 0 ? String.valueOf(serialVersionID) : String.valueOf(environment.getSerialVersionID());
                String argLabel = buildEvent(environment, task);

                System.out.println("tracking event: [" + lid + ", " + argAction+ "," + argLabel +"]");
                //tracker.trackEvent(lid, argAction, argLabel);
                GoogleAnalyticsRequest request = new GoogleAnalyticsRequest();
/*
                request.userId(lid);
                request.applicationName(environment.getBackendUsed());
                request.applicationVersion("0.4-rc3.9");
                */


          //      PageViewHit hit = new PageViewHit("http://www.nd4j.org/ast", "Direct Hit");


            //    tracker.post(hit);


                System.out.println("Request sent");
            } catch (Exception e) {
                // keep quiet on exceptions here
                ;
                throw new RuntimeException(e);
            } finally {
                ;
            }
        }
    }

    private String buildEvent(Environment environment, Task task) {
        StringBuilder builder = new StringBuilder(environment.toCompactString());
        builder.append(task.toCompactString());

        return builder.toString();
    }
}
