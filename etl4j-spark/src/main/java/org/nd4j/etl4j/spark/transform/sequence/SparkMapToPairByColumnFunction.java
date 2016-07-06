package org.nd4j.etl4j.spark.transform.sequence;

import lombok.AllArgsConstructor;
import org.apache.spark.api.java.function.PairFunction;
import org.canova.api.writable.Writable;
import scala.Tuple2;

import java.util.List;

/**
 * Spark function to map a n example to a pair, by using one of the columns as the key.
 *
 * @author Alex Black
 */
@AllArgsConstructor
public class SparkMapToPairByColumnFunction implements PairFunction<List<Writable>,Writable,List<Writable>> {

    private final int keyColumnIdx;

    @Override
    public Tuple2<Writable, List<Writable>> call(List<Writable> writables) throws Exception {
        return new Tuple2<>(writables.get(keyColumnIdx),writables);
    }
}
