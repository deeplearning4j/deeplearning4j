package org.deeplearning4j.iterativereduce.impl.reader;

import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.RecordReader;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;

/**
 * canova record reader
 * @author Adam Gibson
 */
public class CanovaRecordReader implements RecordReader<Long, Collection<Writable>> {
    private org.canova.api.records.reader.RecordReader recordReader;
    private int numRecords = 0;
    private Collection<Writable> currRecord;

    public CanovaRecordReader(org.canova.api.records.reader.RecordReader recordReader) {
        this.recordReader = recordReader;
    }

    public void initialize(InputSplit inputSplit) throws IOException, InterruptedException {
        recordReader.initialize(new CanovaMapRedInputSplit(inputSplit));
    }




    @Override
    public boolean next(Long aLong, Collection<Writable> writables) throws IOException {
        return recordReader.hasNext();
    }

    @Override
    public Long createKey() {
        return Long.valueOf(numRecords);
    }

    @Override
    public Collection<Writable> createValue() {
        if(recordReader.hasNext()) {
            Collection<org.canova.api.writable.Writable> writables = recordReader.next();
            Collection<Writable> wrapped = new ArrayList<>();
            for(org.canova.api.writable.Writable writable : writables)
                wrapped.add(new CanovaWritableDelegate(writable));
            currRecord = wrapped;


        }

        return currRecord;
    }

    @Override
    public long getPos() throws IOException {
        return 0;
    }

    @Override
    public void close() throws IOException {
        recordReader.close();
    }

    @Override
    public float getProgress() throws IOException {
        return 0;
    }
}
