package org.deeplearning4j.spark.impl.common.repartition;

import lombok.AllArgsConstructor;
import org.apache.spark.api.java.function.Function2;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * Created by Alex on 03/07/2016.
 */
@AllArgsConstructor
public class AssignIndexFunction<T> implements Function2<Integer, Iterator<T>, Iterator<Tuple2<Integer,T>>> {
    private final int[] partitionElementStartIdxs;
    @Override
    public Iterator<Tuple2<Integer, T>> call(Integer partionNum, Iterator<T> v2) throws Exception {
        int currIdx = partitionElementStartIdxs[partionNum];
        List<Tuple2<Integer,T>> list = new ArrayList<>();
        while(v2.hasNext()){
            list.add(new Tuple2<>(currIdx++, v2.next()));
        }
        return list.iterator();
    }
}
