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

package org.nd4j.linalg.heartbeat;


import org.nd4j.linalg.heartbeat.reports.Environment;
import org.nd4j.linalg.heartbeat.reports.Event;
import org.nd4j.linalg.heartbeat.reports.Task;

import java.util.concurrent.atomic.AtomicBoolean;

/**
 *
 * Heartbeat implementation for ND4j
 *
 * @author raver119@gmail.com
 */
public class Heartbeat {
    private static final Heartbeat INSTANCE = new Heartbeat();
    private volatile long serialVersionID;
    private AtomicBoolean enabled = new AtomicBoolean(true);


    protected Heartbeat() {

    }

    public static Heartbeat getInstance() {
        return INSTANCE;
    }

    public void disableHeartbeat() {
        this.enabled.set(false);
    }

    public synchronized void reportEvent(Event event, Environment environment, Task task) {

    }

    public synchronized void derivedId(long id) {

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

        }
    }

}
