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

package org.nd4j.linalg.api.instrumentation;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.executors.ExecutorServiceProvider;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;

/**
 * Collects log entries in memory
 *
 * @author Adam Gibson
 */
public class InMemoryInstrumentation implements Instrumentation {
    private List<LogEntry> entries = Collections.synchronizedList(new ArrayList<LogEntry>());
    private List<DataBufferLogEntry> dataBufferLogEntries =
                    Collections.synchronizedList(new ArrayList<DataBufferLogEntry>());
    private ExecutorService executorService = ExecutorServiceProvider.getExecutorService();
    private Map<String, Collection<LogEntry>> logEntries = new ConcurrentHashMap<>();

    @Override
    public void log(final INDArray toLog, final String status) {
        executorService.submit(new Runnable() {
            @Override
            public void run() {
                LogEntry entry = new LogEntry(toLog, status);
                entries.add(entry);
                //                Collection<LogEntry> logEntries = InMemoryInstrumentation.this.logEntries.get(toLog.id());
                //
                //                if (logEntries == null) {
                //                    logEntries = new CopyOnWriteArrayList<>();
                //                    InMemoryInstrumentation.this.logEntries.put(toLog.id(), logEntries);
                //                }

                //                logEntries.add(entry);
            }
        });



    }

    @Override
    public void log(final DataBuffer buffer, final String status) {
        executorService.submit(new Runnable() {
            @Override
            public void run() {
                dataBufferLogEntries.add(new DataBufferLogEntry(buffer, status));

            }
        });
    }

    @Override
    public void log(final INDArray toLog) {
        executorService.submit(new Runnable() {
            @Override
            public void run() {
                entries.add(new LogEntry(toLog));

            }
        });
    }

    @Override
    public void log(final DataBuffer buffer) {
        executorService.submit(new Runnable() {
            @Override
            public void run() {
                dataBufferLogEntries.add(new DataBufferLogEntry(buffer));

            }
        });
    }

    @Override
    public Collection<LogEntry> getStillAlive() {
        Set<LogEntry> ret = new HashSet<>();
        for (Map.Entry<String, Collection<LogEntry>> s : logEntries.entrySet()) {
            Collection<LogEntry> coll = s.getValue();
            boolean foundDestroyed = false;
            LogEntry created = null;
            for (LogEntry entry : coll) {
                if (entry.getStatus().equals(Instrumentation.DESTROYED)) {
                    foundDestroyed = true;
                }
                if (entry.getStatus().equals(Instrumentation.CREATED)) {
                    created = entry;
                }

            }

            if (!foundDestroyed)
                if (created != null)
                    ret.add(created);
                else
                    throw new IllegalStateException("Unable to add a non created entry");

        }
        return ret;
    }

    @Override
    public Collection<LogEntry> getDestroyed() {
        Set<LogEntry> ret = new HashSet<>();
        for (Map.Entry<String, Collection<LogEntry>> s : logEntries.entrySet()) {
            Collection<LogEntry> coll = s.getValue();
            for (LogEntry entry : coll) {
                if (entry.getStatus().equals(Instrumentation.DESTROYED)) {
                    ret.add(entry);
                }
            }
        }
        return ret;
    }

    @Override
    public boolean isDestroyed(String id) {
        Collection<LogEntry> logged = logEntries.get(id);
        if (logged == null)
            throw new IllegalArgumentException("No key found " + id);
        return logged.size() == 2;
    }

    public List<DataBufferLogEntry> getDataBufferLogEntries() {
        return dataBufferLogEntries;
    }

    public void setDataBufferLogEntries(List<DataBufferLogEntry> dataBufferLogEntries) {
        this.dataBufferLogEntries = dataBufferLogEntries;
    }

    public List<LogEntry> getEntries() {
        return entries;
    }

    public void setEntries(List<LogEntry> entries) {
        this.entries = entries;
    }
}
