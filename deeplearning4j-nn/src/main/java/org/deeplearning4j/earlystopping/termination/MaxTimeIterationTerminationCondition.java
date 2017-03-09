/*-
 *
 *  * Copyright 2015 Skymind,Inc.
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

package org.deeplearning4j.earlystopping.termination;

import java.util.concurrent.TimeUnit;

/**Terminate training based on max time.
 */
public class MaxTimeIterationTerminationCondition implements IterationTerminationCondition {

    private long maxTimeAmount;
    private TimeUnit maxTimeUnit;
    private long initializationTime;
    private long endTime;

    public MaxTimeIterationTerminationCondition(long maxTimeAmount, TimeUnit maxTimeUnit) {
        if (maxTimeAmount <= 0 || maxTimeUnit == null)
            throw new IllegalArgumentException(
                            "Invalid maximum training time: " + "amount = " + maxTimeAmount + " unit = " + maxTimeUnit);
        this.maxTimeAmount = maxTimeAmount;
        this.maxTimeUnit = maxTimeUnit;
    }

    @Override
    public void initialize() {
        initializationTime = System.currentTimeMillis();
        endTime = initializationTime + maxTimeUnit.toMillis(maxTimeAmount);
    }

    @Override
    public boolean terminate(double lastMiniBatchScore) {
        return System.currentTimeMillis() >= endTime;
    }

    @Override
    public String toString() {
        return "MaxTimeIterationTerminationCondition(" + maxTimeAmount + ",unit=" + maxTimeUnit + ")";
    }
}
