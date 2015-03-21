package org.nd4j.linalg.api.instrumentation;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * Collects log entries in memory
 *
 * @author Adam Gibson
 */
public class InMemoryInstrumentation implements Instrumentation {
    private List<LogEntry> entries = new CopyOnWriteArrayList<>();
    private List<DataBufferLogEntry> dataBufferLogEntries = new CopyOnWriteArrayList<>();

    private Map<String, Collection<LogEntry>> logEntries = new ConcurrentHashMap<>();

    @Override
    public void log(INDArray toLog, String status) {
        LogEntry entry = new LogEntry(toLog, status);
        entries.add(entry);
        Collection<LogEntry> logEntries = this.logEntries.get(toLog.id());

        if (logEntries == null) {
            logEntries = new CopyOnWriteArrayList<>();
            this.logEntries.put(toLog.id(), logEntries);
        }

        logEntries.add(entry);

    }

    @Override
    public void log(DataBuffer buffer, String status) {
        dataBufferLogEntries.add(new DataBufferLogEntry(buffer, status));
    }

    @Override
    public void log(INDArray toLog) {
        entries.add(new LogEntry(toLog));
    }

    @Override
    public void log(DataBuffer buffer) {
        dataBufferLogEntries.add(new DataBufferLogEntry(buffer));
    }

    @Override
    public Collection<LogEntry> getStillAlive() {
        Set<LogEntry> ret = new HashSet<>();
        for (String s : logEntries.keySet()) {
            Collection<LogEntry> coll = logEntries.get(s);
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
        for (String s : logEntries.keySet()) {
            Collection<LogEntry> coll = logEntries.get(s);
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
