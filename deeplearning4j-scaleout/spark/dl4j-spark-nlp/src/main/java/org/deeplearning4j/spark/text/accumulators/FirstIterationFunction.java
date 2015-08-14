package org.deeplearning4j.spark.text.accumulators;

import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;
import java.util.Map;

/**
 * @author jeffreytang
 */
public class FirstIterationFunction implements VoidFunction<Pair<List<VocabWord>, Long>> {

    Broadcast<Map<String, Object>> word2vecVarMapBroadcast;

    public FirstIterationFunction(Broadcast<Map<String, Object>> word2vecVarMapBroadcast) {
        this.word2vecVarMapBroadcast = word2vecVarMapBroadcast;
    }

    @Override
    public void call(Pair<List<VocabWord>, Long> pair) {
        Map<String, Object> word2vecVarMap = word2vecVarMapBroadcast.getValue();
        int vectorLength = (int) word2vecVarMap.get("vectorLength");
        boolean useAdaGrad = (boolean) word2vecVarMap.get("useAdaGrad");
        int negative = (int) word2vecVarMap.get("negative");
        int window = (int) word2vecVarMap.get("window");
        double alpha = (double) word2vecVarMap.get("alpha");
        double minAlpha = (double) word2vecVarMap.get("minAlpha");
        int iterations = (int) word2vecVarMap.get("iterations");
        long seed = (long) word2vecVarMap.get("seed");
        Random rng = Nd4j.getRandom();
        rng.setSeed(seed);



    }


    public INDArray setRandomWeightPerSentence(int vectorLength, Random rng) {
        return Nd4j.rand(new int[]{1 ,vectorLength}, rng).subi(0.5).divi(vectorLength);
    }


}
