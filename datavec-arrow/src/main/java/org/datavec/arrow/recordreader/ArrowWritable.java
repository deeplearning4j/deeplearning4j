package org.datavec.arrow.recordreader;

import org.datavec.api.writable.Writable;
import org.datavec.api.writable.WritableType;

import java.io.DataInput;
import java.io.DataOutput;

public class ArrowWritable implements Writable {

    @Override
    public void write(DataOutput out) {

    }

    @Override
    public void readFields(DataInput in) {

    }

    @Override
    public void writeType(DataOutput out) {

    }

    @Override
    public double toDouble() {
        return 0;
    }

    @Override
    public float toFloat() {
        return 0;
    }

    @Override
    public int toInt() {
        return 0;
    }

    @Override
    public long toLong() {
        return 0;
    }

    @Override
    public WritableType getType() {
        return null;
    }
}
