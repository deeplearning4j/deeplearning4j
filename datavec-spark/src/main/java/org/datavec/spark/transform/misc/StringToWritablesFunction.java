package org.datavec.spark.transform.misc;

import lombok.AllArgsConstructor;
import org.apache.spark.api.java.function.Function;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.StringSplit;
import org.datavec.api.writable.Writable;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * Convert a String to a List<Writable> using a DataVec record reader
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
