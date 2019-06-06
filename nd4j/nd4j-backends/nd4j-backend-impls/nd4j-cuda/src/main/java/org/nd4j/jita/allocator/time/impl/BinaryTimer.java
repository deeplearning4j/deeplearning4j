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

package org.nd4j.jita.allocator.time.impl;

import org.nd4j.jita.allocator.time.RateTimer;

import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

/**
 * This is simple implementation of DecayingTimer, it doesn't store any actual information for number of events happened.
 * Just a fact: there were events, or there were no events
 *
 * @author raver119@gmail.com
 */
public class BinaryTimer implements RateTimer {
    private AtomicLong timer;
    private long timeframeMilliseconds;

    public BinaryTimer(long timeframe, TimeUnit timeUnit) {
        timer = new AtomicLong(System.currentTimeMillis());

        timeframeMilliseconds = TimeUnit.MILLISECONDS.convert(timeframe, timeUnit);
    }

    /**
     * This method notifies timer about event
     */
    @Override
    public void triggerEvent() {
        timer.set(System.currentTimeMillis());
    }

    /**
     * This method returns average frequency of events happened within predefined timeframe
     *
     * @return
     */
    @Override
    public double getFrequencyOfEvents() {
        if (isAlive()) {
            return 1;
        } else {
            return 0;
        }
    }

    /**
     * This method returns total number of events happened withing predefined timeframe
     *
     * @return
     */
    @Override
    public long getNumberOfEvents() {
        if (isAlive()) {
            return 1;
        } else {
            return 0;
        }
    }

    protected boolean isAlive() {
        long currentTime = System.currentTimeMillis();

        if (currentTime - timer.get() > timeframeMilliseconds) {
            return false;
        }

        return true;
    }
}
