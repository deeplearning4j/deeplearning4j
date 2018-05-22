package org.datavec.local.transforms;

import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.transform.TransformProcessSequenceRecordReader;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.writable.Writable;

import java.util.Arrays;
import java.util.List;

public class LocalTransformProcessSequenceRecordReader extends TransformProcessSequenceRecordReader {

    public LocalTransformProcessSequenceRecordReader(SequenceRecordReader sequenceRecordReader, TransformProcess transformProcess) {
        super(sequenceRecordReader, transformProcess);
    }

    @Override
    public List<List<Writable>> sequenceRecord() {
        return LocalTransformExecutor.executeSequenceToSequence(Arrays.asList(sequenceRecordReader.nextSequence().getSequenceRecord()),transformProcess
        ).get(0);
    }

    @Override
    public List<List<Writable>> next(int num) {
        return super.next(num);
    }

    @Override
    public List<Writable> next() {
        return super.next();
    }
}
