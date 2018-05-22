/*-
 *  * Copyright 2016 Skymind, Inc.
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
 */

package org.datavec.api.writable;


import com.google.common.math.DoubleMath;
import org.datavec.api.io.WritableComparable;
import org.datavec.api.io.WritableComparator;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

/**
 * Writable for Double values.
 */
public class DoubleWritable implements WritableComparable {

    private double value = 0.0;

    public DoubleWritable() {

    }

    public DoubleWritable(@JsonProperty("value") double value) {
        set(value);
    }

    public void readFields(DataInput in) throws IOException {
        value = in.readDouble();
    }

    @Override
    public void writeType(DataOutput out) throws IOException {
        out.writeShort(WritableType.Double.typeIdx());
    }

    public void write(DataOutput out) throws IOException {
        out.writeDouble(value);
    }

    public void set(double value) {
        this.value = value;
    }

    public double get() {
        return value;
    }

    public boolean fuzzyEquals(Writable o, double tolerance) {
        double other;
        if (o instanceof IntWritable){
            other = ((IntWritable) o).toDouble();
        } else if (o instanceof  LongWritable) {
            other = ((LongWritable) o).toDouble();
        } else if (o instanceof ByteWritable) {
            other = ((ByteWritable) o).toDouble();
        } else if (o instanceof  DoubleWritable) {
            other = ((DoubleWritable) o).toDouble();
        } else if (o instanceof  FloatWritable) {
            other = ((FloatWritable) o).toDouble();
        } else { return false; }
        return DoubleMath.fuzzyEquals(this.value, other, tolerance);
    }

    /**
     * Returns true iff <code>o</code> is a DoubleWritable with the same value.
     */
    public boolean equals(Object o) {
        if (o instanceof  DoubleWritable){
            DoubleWritable other = (DoubleWritable) o;
            return this.value == other.value;
        }
        if (o instanceof FloatWritable){
            FloatWritable other = (FloatWritable) o;
            float thisFloat = (float) this.value;
            return (this.value == thisFloat && other.get() == thisFloat);
        } else {
            return false;
        }
    }

    public int hashCode() {
        long var2 = Double.doubleToLongBits(value);
        return (int)(var2 ^ var2 >>> 32);
    }

    public int compareTo(Object o) {
        DoubleWritable other = (DoubleWritable) o;
        return (value < other.value ? -1 : (value == other.value ? 0 : 1));
    }

    public String toString() {
        return Double.toString(value);
    }

    /** A Comparator optimized for DoubleWritable. */
    public static class Comparator extends WritableComparator {
        public Comparator() {
            super(DoubleWritable.class);
        }

        public int compare(byte[] b1, int s1, int l1, byte[] b2, int s2, int l2) {
            double thisValue = readDouble(b1, s1);
            double thatValue = readDouble(b2, s2);
            return (thisValue < thatValue ? -1 : (thisValue == thatValue ? 0 : 1));
        }
    }

    static { // register this comparator
        WritableComparator.define(DoubleWritable.class, new Comparator());
    }

    @Override
    public double toDouble() {
        return value;
    }

    @Override
    public float toFloat() {
        return (float) value;
    }

    @Override
    public int toInt() {
        return (int) value;
    }

    @Override
    public long toLong() {
        return (long) value;
    }

    @Override
    public WritableType getType() {
        return WritableType.Double;
    }
}
