package org.canova.spark.transform.rank;

import org.apache.spark.api.java.function.Function;
import org.canova.api.io.data.LongWritable;
import org.canova.api.writable.Writable;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.List;

/**
 * A simple helper function for use in executing CalculateSortedRank
 *
 * @author Alex Black
 */
public class UnzipForCalculateSortedRankFunction implements Function<Tuple2<Tuple2<Writable,List<Writable>>,Long>,List<Writable>> {
    @Override
    public List<Writable> call(Tuple2<Tuple2<Writable, List<Writable>>, Long> v1) throws Exception {
        List<Writable> inputWritables = new ArrayList<>(v1._1()._2());
        inputWritables.add(new LongWritable(v1._2()));
        return inputWritables;
    }
}
