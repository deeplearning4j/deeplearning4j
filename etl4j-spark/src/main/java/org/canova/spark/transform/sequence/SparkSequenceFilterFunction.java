package org.canova.spark.transform.sequence;

import lombok.AllArgsConstructor;
import org.apache.spark.api.java.function.Function;
import org.canova.api.writable.Writable;
import io.skymind.echidna.api.filter.Filter;

import java.util.List;

/**
 * Created by Alex on 5/03/2016.
 */
@AllArgsConstructor
public class SparkSequenceFilterFunction implements Function<List<List<Writable>>,Boolean> {

    private final Filter filter;

    @Override
    public Boolean call(List<List<Writable>> v1) throws Exception {
        return !filter.removeSequence(v1);   //Spark: return true to keep example (Filter: return true to remove)
    }
}
