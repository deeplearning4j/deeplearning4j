package org.canova.spark.functions;

import org.apache.spark.api.java.function.Function;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.split.StringSplit;
import org.canova.api.writable.Writable;

import java.util.Collection;

/**
 * LineRecordReaderFunction: Used to map a {@code JavaRDD<String>} to a {@code JavaRDD<Collection<Writable>>}
 * Note that this is most useful with LineRecordReader instances (CSVRecordReader, SVMLightRecordReader, etc)
 *
 * @author Alex Black
 */
public class LineRecordReaderFunction implements Function<String,Collection<Writable>> {
    private final RecordReader recordReader;

    public LineRecordReaderFunction(RecordReader recordReader){
        this.recordReader = recordReader;
    }

    @Override
    public Collection<Writable> call(String s) throws Exception {
        recordReader.initialize(new StringSplit(s));
        return recordReader.next();
    }
}
