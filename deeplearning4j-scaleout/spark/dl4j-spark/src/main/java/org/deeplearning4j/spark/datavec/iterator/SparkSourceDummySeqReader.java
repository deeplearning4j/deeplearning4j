package org.deeplearning4j.spark.datavec.iterator;

import lombok.Data;
import org.datavec.api.records.SequenceRecord;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.writable.Writable;

import java.io.DataInputStream;
import java.io.IOException;
import java.net.URI;
import java.util.List;

@Data
public class SparkSourceDummySeqReader extends SparkSourceDummyReader implements SequenceRecordReader {
    public SparkSourceDummySeqReader(int readerIdx) {
        super(readerIdx);
    }

    @Override
    public List<List<Writable>> sequenceRecord() {
        throw new UnsupportedOperationException();
    }

    @Override
    public List<List<Writable>> sequenceRecord(URI uri, DataInputStream dataInputStream) throws IOException {
        throw new UnsupportedOperationException();
    }

    @Override
    public SequenceRecord nextSequence() {
        throw new UnsupportedOperationException();
    }

    @Override
    public SequenceRecord loadSequenceFromMetaData(RecordMetaData recordMetaData) throws IOException {
        throw new UnsupportedOperationException();
    }

    @Override
    public List<SequenceRecord> loadSequenceFromMetaData(List<RecordMetaData> list) throws IOException {
        throw new UnsupportedOperationException();
    }
}
