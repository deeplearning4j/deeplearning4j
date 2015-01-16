package org.deeplearning4j.iterativereduce.impl.reader;

import org.apache.hadoop.mapreduce.InputSplit;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.net.URI;

/**
 * Canova input split: canova ---> hadoop
 *
 * @author Adam Gibson
 */
public class CanovaInputSplit implements org.canova.api.split.InputSplit {

    private org.apache.hadoop.mapreduce.InputSplit split;
    private URI[] uris;

    public CanovaInputSplit(InputSplit split) {
        this.split = split;
        try {
            String[] locations = split.getLocations();
            uris = new URI[locations.length];
            for(int i = 0; i < locations.length; i++) {
                uris[i] = URI.create(locations[i]);
            }
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
