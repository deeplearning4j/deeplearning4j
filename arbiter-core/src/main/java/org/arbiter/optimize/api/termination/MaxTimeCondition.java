/*
 *
 *  * Copyright 2016 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */
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
