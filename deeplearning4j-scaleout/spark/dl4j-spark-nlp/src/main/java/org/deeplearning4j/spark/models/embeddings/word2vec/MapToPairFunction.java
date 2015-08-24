package org.deeplearning4j.spark.models.embeddings.word2vec;

import org.apache.spark.api.java.function.Function;
import org.deeplearning4j.berkeley.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Map;

/**
 * @author jeffreytang
 */
public class MapToPairFunction implements Function< Map.Entry<Integer, INDArray>, Pair<Integer, INDArray> > {

    @Override
    public Pair<Integer, INDArray> call(Map.Entry<Integer, INDArray> pair) {
        return new Pair<>(pair.getKey(), pair.getValue());
    }
}
