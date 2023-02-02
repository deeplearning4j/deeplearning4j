/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.nd4j.linalg.profiler.data.eventlogger;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.common.primitives.AtomicBoolean;

@Slf4j
public class EventLogger {

    private static EventLogger SINGLETON = new EventLogger();
    private AtomicBoolean enabled = new AtomicBoolean(false);
    protected EventLogger() {}


    /**
     * Returns whether the event logger is enabled or not.
     * @return
     */
    public boolean isEnabled() {
        return enabled.get();
    }
    /**
     * Set enabled.
     * @param enabled whether the logger should be enabled.
     */
    public void setEnabled(boolean enabled) {
        this.enabled.set(enabled);
    }

    /**
     * Log the event with slf4j using INFO.
     * Note that in order to enable this logging
     * configuring your slf4j backend is required.
     * This usually means setting the:
     * org.nd4j.linalg.profiler.data.eventlogger
     * to INFO
     * @param logEvent
     */
    public void log(LogEvent logEvent) {
        if(enabled.get())
            log.info("{},{},{},{},{},{},{}",
                    logEvent.getEventTimeMs(),
                    logEvent.getEventType(),
                    logEvent.getObjectAllocationType(),
                    logEvent.getAssociatedWorkspace(),
                    logEvent.getThreadName(),
                    logEvent.getDataType(),
                    logEvent.getBytes());
    }

    public static EventLogger getInstance() {
        return SINGLETON;
    }


}
