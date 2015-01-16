package org.deeplearning4j.iterativereduce.impl.reader;

import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.TaskAttemptContext;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;

/**
 * canova record reader
 * @author Adam Gibson
 */
public class CanovaRecordReader extends RecordReader<Long, Collection<Writable>> {
    private org.canova.api.records.reader.RecordReader recordReader;
    private int numRecords = 0;
    private Collection<Writable> currRecord;

    public CanovaRecordReader(org.canova.api.records.reader.RecordReader recordReader) {
        this.recordReader = recordReader;
    }

    @Override
    public void initialize(InputSplit inputSplit, TaskAttemptContext taskAttemptContext) throws IOException, InterruptedException {
        recordReader.initialize(new CanovaInputSplit(inputSplit));
    }

    @Override
    public boolean nextKeyValue() throws IOException, InterruptedException {
        if(recordReader.hasNext()) {
            Collection<org.canova.api.writable.Writable> writables = recordReader.next();
            Collection<Writable> wrapped = new ArrayList<>();
            for(org.canova.api.writable.Writable writable : writables)
                wrapped.add(new CanovaWritableDelegate(writable));
            currRecord = wrapped;
            return recordReader.hasNext();

        }
        return recordReader.hasNext();
    }

    @Override
    public Long getCurrentKey() throws IOException, InterruptedException {
        return Long.valueOf(numRecords);
    }

    @Override
    public Collection<Writable> getCurrentValue() throws IOException, InterruptedException {
        return currRecord;
    }

    @Override
    public float getProgress() throws IOException, InterruptedException {
        return 0;
    }

    @Override
    public void close() throws IOException {
        recordReader.close();
    }
}
