package org.deeplearning4j.perf.listener;

import oshi.json.SystemInfo;

import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

public class SystemPolling {

    private ScheduledExecutorService scheduledExecutorService;
    private long pollEveryMillis;

    private SystemPolling(long pollEveryMillis) {
        this.pollEveryMillis = pollEveryMillis;
    }


    public void run() {
        scheduledExecutorService = Executors.newScheduledThreadPool(1);
        scheduledExecutorService.schedule(new Runnable() {
            @Override
            public void run() {
                SystemInfo systemInfo = new SystemInfo();


            }
        },pollEveryMillis, TimeUnit.MILLISECONDS);
    }




    public static class Builder {
        private long pollEveryMillis;

        public Builder pollEveryMillis(long pollEveryMillis) {
            this.pollEveryMillis = pollEveryMillis;
            return this;
        }


        public SystemPolling build() {
            return new SystemPolling(pollEveryMillis);
        }

    }

}
