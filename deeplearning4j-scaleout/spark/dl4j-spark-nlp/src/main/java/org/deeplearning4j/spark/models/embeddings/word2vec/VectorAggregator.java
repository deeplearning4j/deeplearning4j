package org.deeplearning4j.spark.models.embeddings.word2vec;

import org.apache.spark.api.java.function.Function2;
import org.deeplearning4j.berkeley.CounterMap;
import org.deeplearning4j.berkeley.Pair;

import java.util.Iterator;

/**
 * @author Adam Gibson
 */
public class VectorAggregator implements Function2<CounterMap<Integer,Integer>,Word2VecParam,Word2VecParam> {
    @Override
    public Word2VecParam call(CounterMap<Integer, Integer> counterMap, Word2VecParam word2VecParam) throws Exception {
        Iterator<Pair<Integer,Integer>> iter = counterMap.getPairIterator();
        while(iter.hasNext()) {
            Pair<Integer,Integer> next = iter.next();


        }
        return null;
    }
}
