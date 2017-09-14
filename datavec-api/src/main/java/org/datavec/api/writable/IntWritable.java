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

/** A WritableComparable for ints. */
public class IntWritable implements WritableComparable {

    private int value;

    public IntWritable() {}

    public IntWritable(@JsonProperty("value") int value) {
        set(value);
    }

    /** Set the value of this IntWritable. */
    public void set(int value) {
        this.value = value;
    }

    /** Return the value of this IntWritable. */
    public int get() {
        return value;
    }

    public void readFields(DataInput in) throws IOException {
        value = in.readInt();
    }

    @Override
    public void writeType(DataOutput out) throws IOException {
        out.writeShort(WritableType.Int.typeIdx());
    }

    public void write(DataOutput out) throws IOException {
        out.writeInt(value);
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

    /** Returns true iff <code>o</code> is a IntWritable with the same value. */
    public boolean equals(Object o) {
        if (o instanceof  ByteWritable){
            ByteWritable other = (ByteWritable) o;
            return  this.value == other.get();
        }
        if (o instanceof IntWritable) {
            IntWritable other = (IntWritable) o;
            return this.value == other.get();
        }
        if (o instanceof LongWritable) {
            LongWritable other = (LongWritable) o;
            return this.value == other.get();
        } else {
            return false;
        }
    }

    public int hashCode() {
        return value;
    }

    /** Compares two IntWritables. */
    public int compareTo(Object o) {
        int thisValue = this.value;
        int thatValue = ((IntWritable) o).value;
        return (thisValue < thatValue ? -1 : (thisValue == thatValue ? 0 : 1));
    }

    public String toString() {
        return Integer.toString(value);
    }

    /** A Comparator optimized for IntWritable. */
    public static class Comparator extends WritableComparator {
        public Comparator() {
            super(IntWritable.class);
        }

        public int compare(byte[] b1, int s1, int l1, byte[] b2, int s2, int l2) {
            int thisValue = readInt(b1, s1);
            int thatValue = readInt(b2, s2);
            return (thisValue < thatValue ? -1 : (thisValue == thatValue ? 0 : 1));
        }
    }

    static { // register this comparator
        WritableComparator.define(IntWritable.class, new Comparator());
    }

    @Override
    public double toDouble() {
        return value;
    }

    @Override
    public float toFloat() {
        return value;
    }

    @Override
    public int toInt() {
        return value;
    }

    @Override
    public long toLong() {
        return value;
    }

    @Override
    public WritableType getType() {
        return WritableType.Int;
    }
}
