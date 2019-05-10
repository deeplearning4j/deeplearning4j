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


import org.datavec.api.io.WritableComparable;
import org.datavec.api.io.WritableComparator;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

/**
 * A WritableComparable for booleans. 
 */
public class BooleanWritable implements WritableComparable {

    private boolean value;

    /**
     */
    public BooleanWritable() {};

    /**
     */
    public BooleanWritable(@JsonProperty("value") boolean value) {
        set(value);
    }

    /**
     * Set the value of the BooleanWritable
     */
    public void set(boolean value) {
        this.value = value;
    }

    /**
     * Returns the value of the BooleanWritable
     */
    public boolean get() {
        return value;
    }

    /**
     */
    public void readFields(DataInput in) throws IOException {
        value = in.readBoolean();
    }

    @Override
    public void writeType(DataOutput out) throws IOException {
        out.writeShort(WritableType.Boolean.typeIdx());
    }

    /**
     */
    public void write(DataOutput out) throws IOException {
        out.writeBoolean(value);
    }

    /**
     */
    public boolean equals(Object o) {
        if (!(o instanceof BooleanWritable)) {
            return false;
        }
        BooleanWritable other = (BooleanWritable) o;
        return this.value == other.value;
    }

    public int hashCode() {
        return value ? 0 : 1;
    }



    /**
     */
    public int compareTo(Object o) {
        boolean a = this.value;
        boolean b = ((BooleanWritable) o).value;
        return ((a == b) ? 0 : (a == false) ? -1 : 1);
    }

    public String toString() {
        return Boolean.toString(get());
    }

    /**
     * A Comparator optimized for BooleanWritable.
     */
    public static class Comparator extends WritableComparator {
        public Comparator() {
            super(BooleanWritable.class);
        }

        public int compare(byte[] b1, int s1, int l1, byte[] b2, int s2, int l2) {
            return compareBytes(b1, s1, l1, b2, s2, l2);
        }
    }


    static {
        WritableComparator.define(BooleanWritable.class, new Comparator());
    }

    @Override
    public double toDouble() {
        return (value ? 1.0 : 0.0);
    }

    @Override
    public float toFloat() {
        return (value ? 1.0f : 0.0f);
    }

    @Override
    public int toInt() {
        return (value ? 1 : 0);
    }

    @Override
    public long toLong() {
        return (value ? 1L : 0L);
    }

    @Override
    public WritableType getType() {
        return WritableType.Boolean;
    }
}
