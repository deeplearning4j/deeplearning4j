package org.datavec.api.records.reader.impl.csv;

import org.datavec.api.records.Record;
import org.datavec.api.records.SequenceRecord;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.writable.Writable;

import java.io.DataInputStream;
import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * CSVLineSequenceRecordReader: Used for loading <b>univariance</b> (single valued) sequences from a CSV,
 * where each line in a CSV represents an independent sequence, and each sequence has exactly 1 value
 * per time step.<br>
 * For example, a CSV file with content:
 * <pre>
 * a,b,c
 * 1,2,3,4
 * </pre>
 * will produce two sequences, both with one value per time step; one of length 3 (values a, b, then c for the 3 time steps
 * respectively) and one of length 4 (values 1, 2, 3, then 4 for each of the 4 time steps respectively)
 *
 *
 * @author Alex Black
 */
public class CSVLineSequenceRecordReader extends CSVRecordReader implements SequenceRecordReader {

    /**
     * Default settings: skip 0 lines, use ',' as the delimiter, and '"' for quotes
     */
    public CSVLineSequenceRecordReader(){
        this(0, DEFAULT_DELIMITER, DEFAULT_QUOTE);
    }

    /**
     * Skip lines and use delimiter
     * @param skipNumLines the number of lines to skip
     * @param delimiter the delimiter
     */
    public CSVLineSequenceRecordReader(int skipNumLines, char delimiter) {
        this(skipNumLines, delimiter, '\"');
    }

    /**
     * Skip lines, use delimiter, and strip quotes
     * @param skipNumLines the number of lines to skip
     * @param delimiter the delimiter
     * @param quote the quote to strip
     */
    public CSVLineSequenceRecordReader(int skipNumLines, char delimiter, char quote) {
        super(skipNumLines, delimiter, quote);
    }

    @Override
    public List<List<Writable>> sequenceRecord() {
        return nextSequence().getSequenceRecord();
    }

    @Override
    public List<List<Writable>> sequenceRecord(URI uri, DataInputStream dataInputStream) throws IOException {
        List<Writable> l = record(uri, dataInputStream);
        List<List<Writable>> out = new ArrayList<>();
        for(Writable w : l){
            out.add(Collections.singletonList(w));
        }
        return out;
    }

    @Override
    public SequenceRecord nextSequence() {
        return convert(super.nextRecord());
    }

    @Override
    public SequenceRecord loadSequenceFromMetaData(RecordMetaData recordMetaData) throws IOException {
        return convert(super.loadFromMetaData(recordMetaData));
    }

    @Override
    public List<SequenceRecord> loadSequenceFromMetaData(List<RecordMetaData> recordMetaDatas) throws IOException {
        List<Record> toConvert = super.loadFromMetaData(recordMetaDatas);
        List<SequenceRecord> out = new ArrayList<>();
        for(Record r  : toConvert){
            out.add(convert(r));
        }
        return out;
    }

    protected SequenceRecord convert(Record r){
        List<Writable> line = r.getRecord();
        List<List<Writable>> out = new ArrayList<>();
        for(Writable w : line){
            out.add(Collections.singletonList(w));
        }
        return new org.datavec.api.records.impl.SequenceRecord(out, r.getMetaData());
    }
}
