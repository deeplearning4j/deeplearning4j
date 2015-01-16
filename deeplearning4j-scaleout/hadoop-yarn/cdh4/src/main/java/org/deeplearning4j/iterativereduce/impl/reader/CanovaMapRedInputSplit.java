package org.deeplearning4j.iterativereduce.impl.reader;

import org.apache.hadoop.mapred.FileSplit;
import org.apache.hadoop.mapred.InputSplit;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.net.URI;

/**
 * Canova input split: canova ---> hadoop
 *
 * @author Adam Gibson
 */
public class CanovaMapRedInputSplit implements org.canova.api.split.InputSplit {

    private InputSplit split;
    private URI[] uris;

    public CanovaMapRedInputSplit(InputSplit split) {
        this.split = split;
        try {
            FileSplit split2 = (FileSplit) split;
            //create from the path
            uris = new URI[1];
            uris[0] = URI.create(split2.getPath().toString());
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

    }

    @Override
    public long length() {
        try {
            return split.getLength();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public URI[] locations() {
        return uris;

    }

    @Override
    public void write(DataOutput out) throws IOException {
    }

    @Override
    public void readFields(DataInput in) throws IOException {

    }
}
