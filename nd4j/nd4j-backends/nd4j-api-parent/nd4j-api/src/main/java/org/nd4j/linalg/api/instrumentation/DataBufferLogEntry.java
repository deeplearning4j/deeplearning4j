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
 *
 */

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
    protected long timestamp;
    protected String status = "created";

    public DataBufferLogEntry() {}

    public DataBufferLogEntry(DataBuffer buffer, String status) {
        this.length = buffer.length();
        this.dataType = buffer.dataType() == DataBuffer.Type.DOUBLE ? "double" : "float";
        this.stackTraceElements = Thread.currentThread().getStackTrace();
        this.references = buffer.references();
        timestamp = System.currentTimeMillis();
        this.status = status;
    }

    public DataBufferLogEntry(DataBuffer buffer) {
        this(buffer, "created");
    }


    @Override
    public String toString() {
        return "DataBufferLogEntry{" + "length=" + length + ", references=" + references + ", dataType='" + dataType
                        + '\'' + ", stackTraceElements=" + Arrays.toString(stackTraceElements) + '}';
    }

    public String getStatus() {
        return status;
    }

    public void setStatus(String status) {
        this.status = status;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (!(o instanceof DataBufferLogEntry))
            return false;

        DataBufferLogEntry that = (DataBufferLogEntry) o;

        if (length != that.length)
            return false;
        if (timestamp != that.timestamp)
            return false;
        if (dataType != null ? !dataType.equals(that.dataType) : that.dataType != null)
            return false;
        if (references != null ? !references.equals(that.references) : that.references != null)
            return false;
        if (!Arrays.equals(stackTraceElements, that.stackTraceElements))
            return false;
        if (status != null ? !status.equals(that.status) : that.status != null)
            return false;

        return true;
    }

    @Override
    public int hashCode() {
        int result = (int) (length ^ (length >>> 32));
        result = 31 * result + (references != null ? references.hashCode() : 0);
        result = 31 * result + (dataType != null ? dataType.hashCode() : 0);
        result = 31 * result + (stackTraceElements != null ? Arrays.hashCode(stackTraceElements) : 0);
        result = 31 * result + (int) (timestamp ^ (timestamp >>> 32));
        result = 31 * result + (status != null ? status.hashCode() : 0);
        return result;
    }

    public long getTimestamp() {
        return timestamp;
    }

    public void setTimestamp(long timestamp) {
        this.timestamp = timestamp;
    }

    public long length() {
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
