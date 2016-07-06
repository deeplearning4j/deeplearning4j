package org.canova.api.io.data;

import org.canova.api.io.WritableComparable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

/**
 * NullWritable. Typically only used in very limited circumstances, to signify that a value is missing.
 * Attempts to convert the NullWritable to some other value (using toInt(), toDouble() etc) will result in an
 * UnsupportedOperationException being thrown
 */
public class NullWritable implements WritableComparable{

    public static final NullWritable INSTANCE = new NullWritable();


    @Override
    public int compareTo(Object o) {
        if(this == o) return 0;
        if(!(o instanceof NullWritable)) throw new IllegalArgumentException("Cannot compare NullWritable to " + o.getClass());
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
    public String toString(){
        return "NullWritable";
    }
}
