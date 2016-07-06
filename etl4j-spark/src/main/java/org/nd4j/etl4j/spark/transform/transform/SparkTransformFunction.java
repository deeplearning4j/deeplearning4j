package org.nd4j.etl4j.spark.transform.transform;

import lombok.AllArgsConstructor;
import org.apache.spark.api.java.function.Function;
import org.nd4j.etl4j.api.writable.Writable;
import org.nd4j.etl4j.api.transform.Transform;

import java.util.List;

/**
 * Created by Alex on 5/03/2016.
 */
@AllArgsConstructor
public class SparkTransformFunction implements Function<List<Writable>,List<Writable>> {

    private final Transform transform;

    @Override
    public List<Writable> call(List<Writable> v1) throws Exception {
        return transform.map(v1);
    }
}
