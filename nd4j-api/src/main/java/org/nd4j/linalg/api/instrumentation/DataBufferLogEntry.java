package org.nd4j.linalg.api.instrumentation;

import org.nd4j.linalg.api.buffer.DataBuffer;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Collection;

/**
 * Data buffer log entry. Used for tracking buffer statistics
 * 
 * @author Adam Gibson
 */
public class DataBufferLogEntry implements Serializable {
    protected long length;
    protected Collection<String> references;
    protected String dataType;
    protected StackTraceElement[] stackTraceElements;

    public DataBufferLogEntry(DataBuffer buffer) {
        this.length = buffer.length();
        this.dataType = buffer.dataType() == DataBuffer.DOUBLE ? "double" : "float";
        this.stackTraceElements = Thread.currentThread().getStackTrace();
        this.references = buffer.references();
    }

    @Override
    public String toString() {
        return "DataBufferLogEntry{" +
                "length=" + length +
                ", references=" + references +
                ", dataType='" + dataType + '\'' +
                ", stackTraceElements=" + Arrays.toString(stackTraceElements) +
                '}';
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof DataBufferLogEntry)) return false;

        DataBufferLogEntry that = (DataBufferLogEntry) o;

        if (length != that.length) return false;
        if (dataType != null ? !dataType.equals(that.dataType) : that.dataType != null) return false;
        if (references != null ? !references.equals(that.references) : that.references != null)
            return false;
        if (!Arrays.equals(stackTraceElements, that.stackTraceElements)) return false;

        return true;
    }

    @Override
    public int hashCode() {
        int result = (int) (length ^ (length >>> 32));
        result = 31 * result + (references != null ? references.hashCode() : 0);
        result = 31 * result + (dataType != null ? dataType.hashCode() : 0);
        result = 31 * result + (stackTraceElements != null ? Arrays.hashCode(stackTraceElements) : 0);
        return result;
    }

    public long getLength() {
        return length;
    }

    public void setLength(long length) {
        this.length = length;
    }


    public Collection<String> getReferences() {
        return references;
    }

    public void setReferences(Collection<String> references) {
        this.references = references;
    }

    public String getDataType() {
        return dataType;
    }

    public void setDataType(String dataType) {
        this.dataType = dataType;
    }

    public StackTraceElement[] getStackTraceElements() {
        return stackTraceElements;
    }

    public void setStackTraceElements(StackTraceElement[] stackTraceElements) {
        this.stackTraceElements = stackTraceElements;
    }
}
