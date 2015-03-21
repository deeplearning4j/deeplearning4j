package org.nd4j.linalg.api.instrumentation;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * Collects log entries in memory
 * @author Adam Gibson
 */
public class InMemoryInstrumentation implements Instrumentation {
    private List<LogEntry> entries = new CopyOnWriteArrayList<>();
    private List<DataBufferLogEntry> dataBufferLogEntries = new CopyOnWriteArrayList<>();

    @Override
    public void log(INDArray toLog, String status) {
        entries.add(new LogEntry(toLog,status));
    }

    @Override
    public void log(DataBuffer buffer, String status) {
        dataBufferLogEntries.add(new DataBufferLogEntry(buffer,status));
    }

    @Override
    public void log(INDArray toLog) {
        entries.add(new LogEntry(toLog));
    }

    @Override
    public void log(DataBuffer buffer) {
        dataBufferLogEntries.add(new DataBufferLogEntry(buffer));
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
