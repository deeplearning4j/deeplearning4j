package org.nd4j.linalg.heartbeat;

import com.brsanthu.googleanalytics.EventHit;
import com.brsanthu.googleanalytics.GoogleAnalytics;
import com.brsanthu.googleanalytics.GoogleAnalyticsRequest;
import com.brsanthu.googleanalytics.PageViewHit;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.heartbeat.reports.Environment;
import org.nd4j.linalg.heartbeat.reports.Event;
import org.nd4j.linalg.heartbeat.reports.Task;
import org.nd4j.linalg.heartbeat.utils.EnvironmentUtils;
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
            tracker = new GoogleAnalytics("UA-48811288-4", "nd4j", "rc3.9-SNAPSHOT");
        } catch (Exception e) {
            ;
        }

        try {
            cId = EnvironmentUtils.buildCId();
        } catch (Exception e) {
            ;
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
        // set to one message per hour for now
        if (lastReport.get() < currentTime - (60 * 60 * 1000)) {
            lastReport.set(currentTime);

            RepoThread thread = new RepoThread(event, environment, task);
            thread.start();
        };// else System.out.println("Skipping report");
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
                String argAction = serialVersionID != 0 ? String.valueOf(serialVersionID) : String.valueOf(cId);



                EventHit eventHit = buildEvent(event, environment, task);
                eventHit.userId(argAction);
                eventHit.userAgent(environment.toCompactString());

                tracker.post(eventHit);
            } catch (Exception e) {
                // keep quiet on exceptions here
                ;
                throw new RuntimeException(e);
            } finally {
                ;
            }
        }
    }

    private EventHit buildEvent(Event event, Environment environment, Task task) {
        /*
            So, here's the mapping:
            event category: either standalone/spark/export

            event action: NetworkType/ArchitectureType

            event label: static > Total Memory GB

            event value: size of memory in GB
         */
        String eventAction = task.getNetworkType() + "/" + task.getArchitectureType();
        int mm = Math.max((int) (environment.getAvailableMemory() / 1024 / 1024 / 1024), 1);

        EventHit hit  = new EventHit(event.toString(), eventAction,  "Total memory (GB)", mm );

        //System.out.println("Reporting: {"+ event.toString()+", " + eventAction + ", " +mm+ "}" );
        //System.out.println("Environment: " + environment.toCompactString());

        return hit;
    }
}
