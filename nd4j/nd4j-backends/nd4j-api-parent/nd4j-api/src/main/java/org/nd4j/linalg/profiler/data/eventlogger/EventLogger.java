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
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.common.primitives.AtomicBoolean;
import org.nd4j.linalg.api.memory.Deallocator;
import org.nd4j.linalg.profiler.UnifiedProfiler;

import java.io.PrintStream;
import java.sql.Timestamp;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * EventLogger is used for profiling allocations, deallocations and other events
 * in nd4j. The logger can be turned on using:
 * {@link #setEnabled(boolean)} to true
 * Note that when turning this on there might be slight overhead when de allocating objects.
 *
 * In order to track deallocation events all {@link org.nd4j.linalg.api.memory.Deallocator}
 * have an associated {@link Deallocator#logEvent()}
 * that is null when the {@link EventLogger} is enabled.
 *
 * This is due to us needing to avoid having a reference to the object we're deallocating but instead
 * just retaining relevant metadata.
 *
 * Please only turn this on when debugging something that is hard to track down such as deallocations.
 *
 * Note for ease of use we also exposed this in {@link org.nd4j.linalg.profiler.UnifiedProfiler#enableLogEvents(boolean)}
 * The logger is also automatically turned on when using {@link UnifiedProfiler#start()}
 * and disabled with {@link UnifiedProfiler#stop()}
 *
 * Of note when logging events, all event types of:
 * {@link EventType#values()} can be found in {@link #getEventTypesToLog()}
 * and {@link ObjectAllocationType#values()} cna be found in {@link #getAllocationTypesToLog()}
 * by default.
 *
 * If you want to filter what types of events or allocation types get logged then please call
 * {@link #setAllocationTypesToLog(List)} and {@link #setEventTypesToLog(List)}
 * to configure the filtering.
 *
 * The user can also configure the output. If a user wants to configure which PrintStream
 * to print to, a user can call {@link #setLogStream(PrintStream)}
 *
 * If you want to format the output to use dates instead of nanoseconds by default call
 * {@link #setFormatTimeAsDate(boolean)} to true
 *
 * @author Adam Gibson
 */
@Slf4j
public class EventLogger {

    private PrintStream logStream = System.err;
    private static EventLogger SINGLETON = new EventLogger();
    private AtomicBoolean enabled = new AtomicBoolean(Boolean.parseBoolean(System.getProperty(ND4JSystemProperties.EVENT_LOGGER_ENABLED,"false")));

    private AtomicBoolean formatTimeAsDate = new AtomicBoolean(Boolean.parseBoolean(System.getProperty(ND4JSystemProperties.EVENT_LOGGER_FORMAT_AS_DATE,"false")));

    private List<ObjectAllocationType> allocationTypesToLog = new ArrayList<>(
            Arrays.asList(ObjectAllocationType.values())
    );

    private List<EventType> eventTypesToLog = new ArrayList<>(
            Arrays.asList(EventType.values())
    );


    private List<EventLogListener> listeners = new ArrayList<>();

    protected EventLogger() {}


    public boolean getFormatTimeAsDate() {
        return formatTimeAsDate.get();
    }

    public void setFormatTimeAsDate(boolean formatTimeAsDate) {
        this.formatTimeAsDate.set(formatTimeAsDate);
    }

    /**
     * Clear the listeners from the {@link EventLogger}
     */
    public void clearListeners() {
        listeners.clear();
    }


    /**
     * Remove the specified listener from the {@link EventLogger}
     * @param logListener the log listener to remove
     */
    public void removeListener(EventLogListener logListener) {
        listeners.remove(logListener);
    }

    /**
     * Add a listener to the {@link EventLogger}
     * @param logListener the log listener to add
     */
    public void addEventLogListener(EventLogListener logListener) {
        listeners.add(logListener);
    }

    public PrintStream getLogStream() {
        return logStream;
    }

    public void setLogStream(PrintStream logStream) {
        this.logStream = logStream;
    }

    /**
     * Get the allocation types to log.
     * Defaults to all of them.
     * @return
     */
    public List<ObjectAllocationType> getAllocationTypesToLog() {
        return allocationTypesToLog;
    }

    /**
     * Set the allocation types to log.
     * Overriding this affects what events
     * with what {@link ObjectAllocationType} get logged.
     * @param allocationTypesToLog
     */
    public void setAllocationTypesToLog(List<ObjectAllocationType> allocationTypesToLog) {
        this.allocationTypesToLog = allocationTypesToLog;
    }

    /**
     * Get the event types to log.
     * Only {@link EventType}'s contained
     * in this list will be logged.
     * @return
     */
    public List<EventType> getEventTypesToLog() {
        return eventTypesToLog;
    }

    /**
     * Set the event types to log.
     * Only event types contained in this list will
     * be logged.
     * @param eventTypesToLog
     */
    public void setEventTypesToLog(List<EventType> eventTypesToLog) {
        this.eventTypesToLog = eventTypesToLog;
    }

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
     * Log the event to the specified {@link PrintStream}
     * defaulting to {@link System#err}
     * This usually means setting the:
     * org.nd4j.linalg.profiler.data.eventlogger
     * to TRACE
     *
     * @param logEvent the log event to log.
     */
    public void log(LogEvent logEvent) {
        if(enabled.get() && eventTypesToLog.contains(logEvent.getEventType()) &&
                this.allocationTypesToLog.contains(logEvent.getObjectAllocationType()))
            logStream.println(String.format("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s",
                    formatTimeAsDate.get() ?  new Timestamp(logEvent.getEventTimeMs()) : logEvent.getEventTimeMs(),
                    logEvent.getEventType(),
                    logEvent.getObjectAllocationType(),
                    logEvent.getAssociatedWorkspace(),
                    logEvent.getThreadName(),
                    logEvent.getDataType(),
                    logEvent.getBytes(),
                    logEvent.isAttached(),
                    logEvent.isConstant(),
                    logEvent.getObjectId()));

        if(listeners != null) {
            for(EventLogListener listener : listeners) {
                if(listener != null) {
                    listener.onLogEvent(logEvent);
                }
            }
        }

    }



    public static EventLogger getInstance() {
        return SINGLETON;
    }


}
