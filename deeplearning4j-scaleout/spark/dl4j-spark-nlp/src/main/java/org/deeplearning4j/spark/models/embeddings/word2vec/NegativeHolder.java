package org.deeplearning4j.spark.models.embeddings.word2vec;

import lombok.Getter;
import lombok.NonNull;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.nd4j.linalg.api.buffer.FloatBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.Serializable;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 *
 * Simple singleton holder class for w2v negative sampling, to avoid syn1Neg creation for each spark node
 *
 * @author raver119@gmail.com
 */
public class NegativeHolder implements Serializable {
    private static NegativeHolder ourInstance = new NegativeHolder();

    public static NegativeHolder getInstance() {
        return ourInstance;
    }

    @Getter private volatile INDArray syn1Neg;
    @Getter private volatile INDArray table;

    private transient AtomicBoolean wasInit = new AtomicBoolean(false);
    private transient VocabCache<VocabWord> vocab;

    private NegativeHolder() {

    }

    public synchronized void initHolder(@NonNull VocabCache<VocabWord> vocabCache, double[] expTable, int layerSize) {
         if (!wasInit.get()) {
            this.vocab = vocabCache;
            this.syn1Neg = Nd4j.zeros(vocabCache.numWords(), layerSize);
            makeTable(Math.max(expTable.length, 100000), 0.75);
            wasInit.set(true);
        }
    }
    protected void makeTable(int tableSize,double power) {
        int vocabSize = vocab.numWords();
        table = Nd4j.create(new FloatBuffer(tableSize));
        double trainWordsPow = 0.0;
        for(String word : vocab.words()) {
            trainWordsPow += Math.pow(vocab.wordFrequency(word), power);
        }

        int wordIdx = 0;
        String word = vocab.wordAtIndex(wordIdx);
        double d1 = Math.pow(vocab.wordFrequency(word),power) / trainWordsPow;
        for(int i = 0; i < tableSize; i++) {
            table.putScalar(i,wordIdx);
            double mul = i * 1.0 / (double) tableSize;
            if(mul > d1) {
                if( wordIdx < vocabSize-1 )
                    wordIdx++;
                word = vocab.wordAtIndex(wordIdx);
                String wordAtIndex = vocab.wordAtIndex(wordIdx);
                if(word == null)
                    continue;
                d1 += Math.pow(vocab.wordFrequency(wordAtIndex),power) / trainWordsPow;
            }
        }
    }


}
