package org.deeplearning4j.spark.models.embeddings.word2vec;

import org.apache.spark.broadcast.Broadcast;
import org.datavec.spark.transform.BaseFlatMapFunctionAdaptee;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.nd4j.linalg.api.ndarray.INDArray;
import scala.Tuple2;

import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

/**
 * @author jeffreytang
 * @author raver119@gmail.com
 */
public class FirstIterationFunction extends
                BaseFlatMapFunctionAdaptee<Iterator<Tuple2<List<VocabWord>, Long>>, Entry<VocabWord, INDArray>> {

    public FirstIterationFunction(Broadcast<Map<String, Object>> word2vecVarMapBroadcast,
                    Broadcast<double[]> expTableBroadcast, Broadcast<VocabCache<VocabWord>> vocabCacheBroadcast) {
        super(new FirstIterationFunctionAdapter(word2vecVarMapBroadcast, expTableBroadcast, vocabCacheBroadcast));
    }
}

