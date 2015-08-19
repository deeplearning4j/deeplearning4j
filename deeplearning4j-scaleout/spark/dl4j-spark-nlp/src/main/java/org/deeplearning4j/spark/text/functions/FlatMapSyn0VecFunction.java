package org.deeplearning4j.spark.text.functions;

import org.apache.spark.api.java.function.FlatMapFunction;
import org.deeplearning4j.berkeley.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * @author jeffreytang
 */
public class FlatMapSyn0VecFunction
        implements FlatMapFunction< Map.Entry<Integer, List<INDArray>>, Pair<Integer, INDArray> > {

    @Override
    public Iterable<Pair<Integer, INDArray>> call(Map.Entry<Integer, List<INDArray>> pair) {
        Integer nthRow = pair.getKey();
        List<INDArray> syn0VecList = pair.getValue();

        ArrayList<Pair<Integer, INDArray>> pairList = new ArrayList<>();
        for (INDArray syn0Vec : syn0VecList) {
            pairList.add(new Pair<> (nthRow, syn0Vec));
        }
        return pairList;
    }
}
