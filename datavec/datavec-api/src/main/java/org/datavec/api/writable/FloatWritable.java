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

package org.datavec.api.writable;


import com.google.common.math.DoubleMath;
import org.datavec.api.io.WritableComparable;
import org.datavec.api.io.WritableComparator;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

/** A WritableComparable for floats. */
public class FloatWritable implements WritableComparable {

    private float value;

    public FloatWritable() {}

    public FloatWritable(@JsonProperty("value") float value) {
        set(value);
    }

    /** Set the value of this FloatWritable. */
    public void set(float value) {
        this.value = value;
    }

    /** Return the value of this FloatWritable. */
    public float get() {
        return value;
    }

    public void readFields(DataInput in) throws IOException {
        value = in.readFloat();
    }

    @Override
    public void writeType(DataOutput out) throws IOException {
        out.writeShort(WritableType.Float.typeIdx());
    }

    public void write(DataOutput out) throws IOException {
        out.writeFloat(value);
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

    /** Returns true iff <code>o</code> is a FloatWritable with the same value. */
    public boolean equals(Object o) {
        if (o instanceof FloatWritable){
            FloatWritable other = (FloatWritable) o;
            return this.value == other.value;
        }
        if (o instanceof DoubleWritable){
            DoubleWritable other = (DoubleWritable) o;
            float otherFloat = (float) other.get();
            return (other.get() == otherFloat && this.value == otherFloat);
        } else {
            return false;
        }
    }

    public int hashCode() {
        // defer to Float.hashCode, which does what we mean for it to do
        return Float.floatToIntBits(value);
    }

    /** Compares two FloatWritables. */
    public int compareTo(Object o) {
        float thisValue = this.value;
        float thatValue = ((FloatWritable) o).value;
        return (thisValue < thatValue ? -1 : (thisValue == thatValue ? 0 : 1));
    }

    public String toString() {
        return Float.toString(value);
    }

    /** A Comparator optimized for FloatWritable. */
    public static class Comparator extends WritableComparator {
        public Comparator() {
            super(FloatWritable.class);
        }

        public int compare(byte[] b1, int s1, int l1, byte[] b2, int s2, int l2) {
            float thisValue = readFloat(b1, s1);
            float thatValue = readFloat(b2, s2);
            return (thisValue < thatValue ? -1 : (thisValue == thatValue ? 0 : 1));
        }
    }

    static { // register this comparator
        WritableComparator.define(FloatWritable.class, new Comparator());
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
        return (int) value;
    }

    @Override
    public long toLong() {
        return (long) value;
    }

    @Override
    public WritableType getType() {
        return WritableType.Float;
    }
}
