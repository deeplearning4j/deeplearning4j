package org.canova.api.records.reader.impl;

import org.canova.api.conf.Configuration;
import org.canova.api.io.data.Text;
import org.canova.api.records.reader.BaseRecordReader;
import org.canova.api.split.InputSplit;
import org.canova.api.split.ListStringSplit;
import org.canova.api.writable.Writable;

import java.io.DataInputStream;
import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;

/**
 * Iterates through a list of strings return a record.
 * Only accepts an @link {ListStringInputSplit} as input.
 *
 * @author Adam Gibson
 */
public class ListStringRecordReader extends BaseRecordReader {
    private List<List<String>> delimitedData;
    private Iterator<List<String>> dataIter;
    private Configuration conf;
    /**
     * Called once at initialization.
     *
     * @param split the split that defines the range of records to read
     * @throws IOException
     * @throws InterruptedException
     */
    @Override
    public void initialize(InputSplit split) throws IOException, InterruptedException {
        if(split instanceof ListStringSplit) {
            ListStringSplit listStringSplit = (ListStringSplit) split;
            delimitedData = listStringSplit.getData();
            dataIter = delimitedData.iterator();
        }
        else {
            throw new IllegalArgumentException("Illegal type of input split " + split.getClass().getName());
        }
    }

    /**
     * Called once at initialization.
     *
     * @param conf  a configuration for initialization
     * @param split the split that defines the range of records to read
     * @throws IOException
     * @throws InterruptedException
     */
    @Override
    public void initialize(Configuration conf, InputSplit split) throws IOException, InterruptedException {
        initialize(split);
    }

    /**
     * Get the next record
     *
     * @return
     */
    @Override
    public Collection<Writable> next() {
        List<String> next = dataIter.next();
        invokeListeners(next);
        List<Writable> ret = new ArrayList<>();
        for(String s : next)
            ret.add(new Text(s));
        return ret;
    }

    /**
     * Whether there are anymore records
     *
     * @return
     */
    @Override
    public boolean hasNext() {
        return dataIter.hasNext();
    }

    /**
     * List of label strings
     *
     * @return
     */
    @Override
    public List<String> getLabels() {
        return null;
    }

    /**
     * Reset record reader iterator
     *
     * @return
     */
    @Override
    public void reset() {
        dataIter = delimitedData.iterator();
    }

    /**
     * Load the record from the given DataInputStream
     * Unlike {@link #next()} the internal state of the RecordReader is not modified
     * Implementations of this method should not close the DataInputStream
     *
     * @param uri
     * @param dataInputStream
     * @throws IOException if error occurs during reading from the input stream
     */
    @Override
    public Collection<Writable> record(URI uri, DataInputStream dataInputStream) throws IOException {
        return null;
    }

    /**
     * Closes this stream and releases any system resources associated
     * with it. If the stream is already closed then invoking this
     * method has no effect.
     * <p>
     * <p> As noted in {@link AutoCloseable#close()}, cases where the
     * close may fail require careful attention. It is strongly advised
     * to relinquish the underlying resources and to internally
     * <em>mark</em> the {@code Closeable} as closed, prior to throwing
     * the {@code IOException}.
     *
     * @throws IOException if an I/O error occurs
     */
    @Override
    public void close() throws IOException {

    }

    /**
     * Set the configuration to be used by this object.
     *
     * @param conf
     */
    @Override
    public void setConf(Configuration conf) {
        this.conf = conf;
    }

    /**
     * Return the configuration used by this object.
     */
    @Override
    public Configuration getConf() {
        return conf;
    }
}
