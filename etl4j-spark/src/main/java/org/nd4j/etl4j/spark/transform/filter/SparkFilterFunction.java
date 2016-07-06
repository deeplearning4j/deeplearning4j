package org.nd4j.etl4j.spark.transform.filter;

import lombok.AllArgsConstructor;
import org.apache.spark.api.java.function.Function;
import org.canova.api.writable.Writable;
import org.nd4j.etl4j.api.transform.filter.Filter;

import java.util.List;

/**
 * Spark function for executing filter operations
 * @author Alex Black
 */
@AllArgsConstructor
public class SparkFilterFunction implements Function<List<Writable>,Boolean> {

    private final Filter filter;

    @Override
    public Boolean call(List<Writable> v1) throws Exception {
        return !filter.removeExample(v1);   //Spark: return true to keep example (Filter: return true to remove)
    }
}
