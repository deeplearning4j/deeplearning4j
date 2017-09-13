package org.deeplearning4j.spark.models.embeddings.word2vec;

import org.apache.spark.api.java.function.Function;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;

import java.util.Map;

/**
 * @author jeffreytang
 */
public class MapToPairFunction implements Function<Map.Entry<VocabWord, INDArray>, Pair<VocabWord, INDArray>> {

    @Override
    public Pair<VocabWord, INDArray> call(Map.Entry<VocabWord, INDArray> pair) {
        return new Pair<>(pair.getKey(), pair.getValue());
    }
}
