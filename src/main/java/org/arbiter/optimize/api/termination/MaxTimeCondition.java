package org.arbiter.optimize.api.termination;

import org.arbiter.optimize.runner.IOptimizationRunner;

import java.util.concurrent.TimeUnit;

public class MaxTimeCondition implements TerminationCondition {

    private long duration;
    private TimeUnit timeUnit;
    private long startTime;
    private long endTime;

    public MaxTimeCondition(long duration, TimeUnit timeUnit) {
        this.duration = duration;
        this.timeUnit = timeUnit;
    }

    @Override
    public void initialize(IOptimizationRunner optimizationRunner) {
        startTime = System.currentTimeMillis();
        this.endTime = startTime + timeUnit.toMillis(duration);
    }

    @Override
    public boolean terminate(IOptimizationRunner optimizationRunner) {
        return System.currentTimeMillis() >= endTime;
    }

    @Override
    public String toString(){
        return "MaxTimeCondition("+duration+","+timeUnit+",start="+startTime+",end="+endTime+")";
    }
}
