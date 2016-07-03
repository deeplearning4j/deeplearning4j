package org.deeplearning4j.spark.impl.common.repartition;

import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.api.java.function.PairFunction;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * Created by Alex on 03/07/2016.
 */
public class MapTupleToPairFlatMap<T,U> implements PairFlatMapFunction<Iterator<Tuple2<T,U>>,T,U> {

    @Override
    public Iterable<Tuple2<T, U>> call(Iterator<Tuple2<T, U>> tuple2Iterator) throws Exception {
        List<Tuple2<T,U>> list = new ArrayList<>();
        while(tuple2Iterator.hasNext()){
            list.add(tuple2Iterator.next());
        }
        return list;
    }
}
