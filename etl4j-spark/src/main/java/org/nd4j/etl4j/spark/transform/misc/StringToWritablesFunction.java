package org.nd4j.etl4j.spark.transform.misc;

import lombok.AllArgsConstructor;
import org.apache.spark.api.java.function.Function;
import org.nd4j.etl4j.api.records.reader.RecordReader;
import org.nd4j.etl4j.api.split.StringSplit;
import org.nd4j.etl4j.api.writable.Writable;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * Convert a String to a List<Writable> using a Canova record reader
 *
 */
@AllArgsConstructor
public class StringToWritablesFunction implements Function<String,List<Writable>> {

    private RecordReader recordReader;

    @Override
    public List<Writable> call(String s) throws Exception {
        recordReader.initialize(new StringSplit(s));
        Collection<Writable> next = recordReader.next();
        if(next instanceof List ) return (List<Writable>)next;
        return new ArrayList<>(next);
    }
}
