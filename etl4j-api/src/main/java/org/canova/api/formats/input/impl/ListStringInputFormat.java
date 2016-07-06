package org.canova.api.formats.input.impl;

import org.canova.api.conf.Configuration;
import org.canova.api.formats.input.InputFormat;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.ListStringRecordReader;
import org.canova.api.split.InputSplit;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

/**
 * Input format for the @link {ListStringRecordReader}
 * @author Adam Gibson
 */
public class ListStringInputFormat implements InputFormat {
    /**
     * Creates a reader from an input split
     *
     * @param split the split to read
     * @param conf
     * @return the reader from the given input split
     */
    @Override
    public RecordReader createReader(InputSplit split, Configuration conf) throws IOException, InterruptedException {
        RecordReader reader = new ListStringRecordReader();
        reader.initialize(conf,split);
        return reader;
    }

    /**
     * Creates a reader from an input split
     *
     * @param split the split to read
     * @return the reader from the given input split
     */
    @Override
    public RecordReader createReader(InputSplit split) throws IOException, InterruptedException {
        RecordReader reader = new ListStringRecordReader();
        reader.initialize(split);
        return reader;
    }

    /**
     * Serialize the fields of this object to <code>out</code>.
     *
     * @param out <code>DataOuput</code> to serialize this object into.
     * @throws IOException
     */
    @Override
    public void write(DataOutput out) throws IOException {

    }

    /**
     * Deserialize the fields of this object from <code>in</code>.
     * <p>
     * <p>For efficiency, implementations should attempt to re-use storage in the
     * existing object where possible.</p>
     *
     * @param in <code>DataInput</code> to deseriablize this object from.
     * @throws IOException
     */
    @Override
    public void readFields(DataInput in) throws IOException {

    }

    /**
     * Convert Writable to double. Whether this is supported depends on the specific writable.
     */
    @Override
    public double toDouble() {
        return 0;
    }

    /**
     * Convert Writable to float. Whether this is supported depends on the specific writable.
     */
    @Override
    public float toFloat() {
        return 0;
    }

    /**
     * Convert Writable to int. Whether this is supported depends on the specific writable.
     */
    @Override
    public int toInt() {
        return 0;
    }

    /**
     * Convert Writable to long. Whether this is supported depends on the specific writable.
     */
    @Override
    public long toLong() {
        return 0;
    }
}
