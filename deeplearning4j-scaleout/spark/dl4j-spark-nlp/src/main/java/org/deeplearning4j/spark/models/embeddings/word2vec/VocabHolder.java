package org.deeplearning4j.spark.models.embeddings.word2vec;

import org.deeplearning4j.models.word2vec.VocabWord;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 *
 * @author raver119@gmail.com
 */
public class VocabHolder {
    private static VocabHolder ourInstance = new VocabHolder();

    private Map<Integer, INDArray> indexSyn0VecMap = new ConcurrentHashMap<>();
    private Map<Integer, INDArray> pointSyn1VecMap = new ConcurrentHashMap<>();
    private HashSet<Long> workers = new LinkedHashSet<>();

    private AtomicLong seed = new AtomicLong(0);
    private AtomicInteger vectorLength = new AtomicInteger(0);

    public static VocabHolder getInstance() {
        return ourInstance;
    }

    private VocabHolder() {

    }

    public void setSeed(long seed, int vectorLength) {
        this.seed.set(seed);
        this.vectorLength.set(vectorLength);
    }

    public INDArray getSyn0Vector(Integer wordIndex) {
        if (!workers.contains(Thread.currentThread().getId()))
            workers.add(Thread.currentThread().getId());

        if (!indexSyn0VecMap.containsKey(wordIndex)) {
            synchronized (this) {
                if (!indexSyn0VecMap.containsKey(wordIndex)) {
                    indexSyn0VecMap.put(wordIndex, getRandomSyn0Vec(vectorLength.get(), wordIndex));
                }
            }
        }

        return indexSyn0VecMap.get(wordIndex);
    }

    public INDArray getSyn1Vector(Integer point) {
        return null;
    }

    private INDArray getRandomSyn0Vec(int vectorLength, long lseed) {
        /*
            we use wordIndex as part of seed here, to guarantee that during word syn0 initialization on dwo distinct nodes, initial weights will be the same for the same word
         */
        return Nd4j.rand(lseed * seed.get(), new int[]{1 ,vectorLength}).subi(0.5).divi(vectorLength);
    }
}
