/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

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
 * Remaining time is estimated from the amount of time for training so far, and the total number of iterations
 * specified by the user
 */
@Slf4j
public class TimeIterationListener extends BaseTrainingListener implements Serializable {
    private long start;
    private int iterationCount;
    private AtomicLong iterationCounter = new AtomicLong(0);

    /**
     * Constructor
     * @param iterationCount The global number of iteration for training (all epochs)
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
