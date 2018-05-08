package org.datavec.local.transforms;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.transform.TransformProcessRecordReader;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.writable.Writable;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * A wrapper around the {@link TransformProcessRecordReader}
 * that uses the {@link LocalTransformExecutor}
 * instead of the {@link TransformProcess} methods.
 *
 * @author Adam Gibson
 */
public class LocalTransformProcessRecordReader extends TransformProcessRecordReader {

    /**
     * Initialize with the internal record reader
     * and the transform process.
     * @param recordReader
     * @param transformProcess
     */
    public LocalTransformProcessRecordReader(RecordReader recordReader, TransformProcess transformProcess) {
        super(recordReader, transformProcess);
    }

//    @Override
//    public List<List<Writable>> next(int num) {
//        return LocalTransformExecutor.execute(recordReader.next(num),transformProcess);
//    }
//
//    @Override
//    public List<Writable> next() {
//        return LocalTransformExecutor.execute(Collections.singletonList(recordReader.next()),transformProcess).get(0);
//    }
}
