/*-
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
package org.deeplearning4j.arbiter.optimize.api.termination;

import lombok.Data;
import lombok.NoArgsConstructor;
import org.deeplearning4j.arbiter.optimize.runner.IOptimizationRunner;
import org.joda.time.DateTimeZone;
import org.joda.time.format.DateTimeFormat;
import org.joda.time.format.DateTimeFormatter;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import javax.annotation.sql.DataSourceDefinitions;
import java.util.concurrent.TimeUnit;

/**
 * Terminate hyperparameter optimization after
 * a fixed amount of time has passed
 * @author Alex Black
 */
@NoArgsConstructor
@Data
public class MaxTimeCondition implements TerminationCondition {

    private static final DateTimeFormatter formatter = DateTimeFormat.forPattern("YYYY-MM-dd HH:mm:ss.SSS zzz").withZone(DateTimeZone.UTC);
    @JsonProperty
    private long duration;
    @JsonProperty
    private TimeUnit timeUnit;
    @JsonProperty
    private long startTime;
    @JsonProperty
    private long endTime;

    /**
     * @param duration Duration of time
     * @param timeUnit Unit that the duration is specified in
     */
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
    public String toString() {
        return "MaxTimeCondition(" + duration + "," + timeUnit + ",start=\"" + formatter.print(startTime) + "\",end=\"" + formatter.print(endTime) + "\")";
    }
}
