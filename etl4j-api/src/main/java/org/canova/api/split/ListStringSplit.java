package org.canova.api.split;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.net.URI;
import java.util.List;

/**
 * An input split that already
 * has delimited data of some kind.
 */
public class ListStringSplit implements InputSplit {
    private List<List<String>> data;


    public ListStringSplit(List<List<String>> data) {
        this.data = data;
    }

    /**
     * Length of the split
     *
     * @return
     */
    @Override
    public long length() {
        return data.size();
    }

    /**
     * Locations of the splits
     *
     * @return
     */
    @Override
    public URI[] locations() {
        return new URI[0];
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
        throw new UnsupportedOperationException();
    }

    /**
     * Convert Writable to float. Whether this is supported depends on the specific writable.
     */
    @Override
    public float toFloat() {
        throw new UnsupportedOperationException();
    }

    /**
     * Convert Writable to int. Whether this is supported depends on the specific writable.
     */
    @Override
    public int toInt() {
        throw new UnsupportedOperationException();
    }

    /**
     * Convert Writable to long. Whether this is supported depends on the specific writable.
     */
    @Override
    public long toLong() {
        throw new UnsupportedOperationException();
    }

    public List<List<String>> getData() {
        return data;
    }
}
