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
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Arrays;
import java.util.Collection;

/**
 * Log entry for statistics about ndarrays
 *
 * @author Adam Gibson
 */
public class LogEntry extends DataBufferLogEntry {

    private String id;
    private long[] shape;
    private long[] stride;
    private String ndArrayType;

    public LogEntry() {}

    public LogEntry(INDArray toLog, String status) {
        //this.id = toLog.id();
        this.shape = toLog.shape();
        this.stride = toLog.stride();
        this.ndArrayType = toLog.getClass().getName();
        this.length = toLog.length();
        this.references = toLog.data().references();
        this.dataType = toLog.data().dataType() == DataBuffer.Type.DOUBLE ? "double" : "float";
        this.timestamp = System.currentTimeMillis();
        this.stackTraceElements = Thread.currentThread().getStackTrace();
        this.status = status;
    }


    public LogEntry(INDArray toLog, StackTraceElement[] stackTraceElements, String status) {
        //this.id = toLog.id();
        this.shape = toLog.shape();
        this.stride = toLog.stride();
        this.ndArrayType = toLog.getClass().getName();
        this.length = toLog.length();
        this.references = toLog.data().references();
        this.dataType = toLog.data().dataType() == DataBuffer.Type.DOUBLE ? "double" : "float";
        this.timestamp = System.currentTimeMillis();
        this.stackTraceElements = stackTraceElements;
        this.status = status;
    }

    public LogEntry(INDArray toLog, StackTraceElement[] stackTraceElements) {
        this(toLog, stackTraceElements, "created");
    }


    public LogEntry(INDArray toLog) {
        this(toLog, Thread.currentThread().getStackTrace());
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (!(o instanceof LogEntry))
            return false;

        LogEntry logEntry = (LogEntry) o;

        if (length != logEntry.length)
            return false;
        if (dataType != null ? !dataType.equals(logEntry.dataType) : logEntry.dataType != null)
            return false;
        if (id != null ? !id.equals(logEntry.id) : logEntry.id != null)
            return false;
        if (ndArrayType != null ? !ndArrayType.equals(logEntry.ndArrayType) : logEntry.ndArrayType != null)
            return false;
        if (references != null ? !references.equals(logEntry.references) : logEntry.references != null)
            return false;
        if (!Arrays.equals(shape, logEntry.shape))
            return false;
        if (!Arrays.equals(stackTraceElements, logEntry.stackTraceElements))
            return false;
        if (!Arrays.equals(stride, logEntry.stride))
            return false;

        return true;
    }

    @Override
    public int hashCode() {
        int result = id != null ? id.hashCode() : 0;
        result = 31 * result + (shape != null ? Arrays.hashCode(shape) : 0);
        result = 31 * result + (stride != null ? Arrays.hashCode(stride) : 0);
        result = 31 * result + (int) (length ^ (length >>> 32));
        result = 31 * result + (ndArrayType != null ? ndArrayType.hashCode() : 0);
        result = 31 * result + (references != null ? references.hashCode() : 0);
        result = 31 * result + (dataType != null ? dataType.hashCode() : 0);
        result = 31 * result + (stackTraceElements != null ? Arrays.hashCode(stackTraceElements) : 0);
        return result;
    }

    @Override
    public String toString() {
        return "LogEntry{" + "id='" + id + '\'' + ", shape=" + Arrays.toString(shape) + ", stride="
                        + Arrays.toString(stride) + ", length=" + length + ", ndArrayType='" + ndArrayType + '\''
                        + ", references=" + references + ", dataType='" + dataType + '\'' + ", stackTraceElements="
                        + Arrays.toString(stackTraceElements) + '}';
    }

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public long[] getShape() {
        return shape;
    }

    public void setShape(long[] shape) {
        this.shape = shape;
    }

    public long[] getStride() {
        return stride;
    }

    public void setStride(long[] stride) {
        this.stride = stride;
    }

    public long length() {
        return length;
    }

    public void setLength(long length) {
        this.length = length;
    }

    public String getNdArrayType() {
        return ndArrayType;
    }

    public void setNdArrayType(String ndArrayType) {
        this.ndArrayType = ndArrayType;
    }

    public Collection<String> getReferences() {
        return references;
    }

    public void setReferences(Collection<String> numReferences) {
        this.references = numReferences;
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
