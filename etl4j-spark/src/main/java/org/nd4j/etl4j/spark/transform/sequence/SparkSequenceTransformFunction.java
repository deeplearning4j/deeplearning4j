package org.nd4j.etl4j.spark.transform.sequence;

import lombok.AllArgsConstructor;
import org.apache.spark.api.java.function.Function;
import org.nd4j.etl4j.api.writable.Writable;
import org.nd4j.etl4j.api.transform.Transform;

import java.util.List;

/**
 * Spark function for transforming sequences using a Transform
 * @author Alex Black
 */
@AllArgsConstructor
public class SparkSequenceTransformFunction implements Function<List<List<Writable>>,List<List<Writable>>> {

    private final Transform transform;

    @Override
    public List<List<Writable>> call(List<List<Writable>> v1) throws Exception {
        return transform.mapSequence(v1);
    }
}
