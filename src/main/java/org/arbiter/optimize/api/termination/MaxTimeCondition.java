package org.arbiter.optimize.api.termination;

import java.util.concurrent.TimeUnit;

public class MaxTimeCondition implements TerminationCondition {

    private double duration;
    private TimeUnit timeUnit;
    private long startTime;
    private long endTime;

    public MaxTimeCondition(long duration, TimeUnit timeUnit) {
        this.startTime = System.currentTimeMillis();
        this.endTime = startTime + timeUnit.toMillis(duration);
    }



    @Override
    public boolean terminate(Object o) {
        return System.currentTimeMillis() > endTime;
    }
}
