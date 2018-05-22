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

import org.datavec.api.io.WritableComparable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

/**
 * NullWritable. Typically only used in very limited circumstances, to signify that a value is missing.
 * Attempts to convert the NullWritable to some other value (using toInt(), toDouble() etc) will result in an
 * UnsupportedOperationException being thrown
 */
public class NullWritable implements WritableComparable {

    public static final NullWritable INSTANCE = new NullWritable();


    @Override
    public int compareTo(Object o) {
        if (this == o)
            return 0;
        if (!(o instanceof NullWritable))
            throw new IllegalArgumentException("Cannot compare NullWritable to " + o.getClass());
        return 0;
    }

    public boolean equals(Object o) {
        return o instanceof NullWritable;
    }

    @Override
    public void write(DataOutput out) throws IOException {
        //No op
    }

    @Override
    public void readFields(DataInput in) throws IOException {
        //No op
    }

    @Override
    public void writeType(DataOutput out) throws IOException {
        out.writeShort(WritableType.Null.typeIdx());
    }

    @Override
    public double toDouble() {
        throw new UnsupportedOperationException("Cannot convert NullWritable to other values");
    }

    @Override
    public float toFloat() {
        throw new UnsupportedOperationException("Cannot convert NullWritable to other values");
    }

    @Override
    public int toInt() {
        throw new UnsupportedOperationException("Cannot convert NullWritable to other values");
    }

    @Override
    public long toLong() {
        throw new UnsupportedOperationException("Cannot convert NullWritable to other values");
    }

    @Override
    public WritableType getType() {
        return WritableType.Null;
    }

    @Override
    public String toString() {
        return "NullWritable";
    }
}
