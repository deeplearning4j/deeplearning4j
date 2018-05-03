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

package org.deeplearning4j.optimize.listeners;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.BaseTrainingListener;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Serializable;
import java.util.Date;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Time Iteration Listener.
 * This listener displays into INFO logs the remaining time in minutes and the date of the end of the process. 
 */
@Slf4j
public class TimeIterationListener extends BaseTrainingListener implements Serializable {
    private long start;
    private int iterationCount;
    private AtomicLong iterationCounter = new AtomicLong(0);

    /**
     * Constructor
     * @param iterationCount The global number of iteration of the process 
     */
    public TimeIterationListener(int iterationCount) {
        this.iterationCount = iterationCount;
        start = System.currentTimeMillis();
    }

    @Override
    public void iterationDone(Model model, int iteration, int epoch) {
        long currentIteration = iterationCounter.incrementAndGet();
        long elapsed = System.currentTimeMillis() - start;
        long remaining = (iterationCount - currentIteration) * elapsed / currentIteration;
        long minutes = remaining / (1000 * 60);
        Date date = new Date(start + elapsed + remaining);
        log.info("Remaining time : " + minutes + "mn - End expected : " + date.toString());
    }

}
