package org.deeplearning4j.models.paragraphvectors.lookuptable;

import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.concurrent.atomic.AtomicLong;

/**
 * Created by agibsonccc on 11/30/14.
 */
public class InMemoryParagraphVectorsLookupTable extends InMemoryLookupTable {



    public InMemoryParagraphVectorsLookupTable(VocabCache vocab, int vectorLength, boolean useAdaGrad, double lr, RandomGenerator gen, double negative) {
        super(vocab, vectorLength, useAdaGrad, lr, gen, negative);
    }

    @Override
    public void iterateSample(VocabWord w1, VocabWord w2, AtomicLong nextRandom, double alpha) {
        if(w2 == null || w2.getIndex() < 0)
            return;
        //current word vector
        INDArray l1 = this.syn0.slice(w2.getIndex());


        //error for current word and context
        INDArray neu1e = Nd4j.create(vectorLength);


        double avgChange = 0.0f;




        for(int i = 0; i < w1.getCodeLength(); i++) {
            int code = w1.getCodes().get(i);
            int point = w1.getPoints().get(i);
            if(point >= syn0.rows() || point < 0)
                throw new IllegalStateException("Illegal point " + point);
            //other word vector

            INDArray syn1 = this.syn1.slice(point);


            double dot = Nd4j.getBlasWrapper().dot(l1,syn1);

            if(dot < -MAX_EXP || dot >= MAX_EXP)
                continue;


            int idx = (int) ((dot + MAX_EXP) * ((double) expTable.length / MAX_EXP / 2.0));
            if(idx >= expTable.length)
                continue;

            //score
            double f =  expTable[idx];
            //gradient
            double g = (1 - code - f) * (useAdaGrad ?  w1.getLearningRate(i,alpha) : alpha);




            avgChange += g;
            if(syn0.data().dataType() == DataBuffer.DOUBLE) {
                Nd4j.getBlasWrapper().axpy(g, syn1, neu1e);
                Nd4j.getBlasWrapper().axpy(g, l1, syn1);
            }
            else {
                Nd4j.getBlasWrapper().axpy((float) g, syn1, neu1e);
                Nd4j.getBlasWrapper().axpy((float) g, l1, syn1);
            }


        }


        int target = w1.getIndex();
        int label;
        //negative sampling
        if(negative > 0)
            for (int d = 0; d < negative + 1; d++) {
                if (d == 0) {

                    label = 1;
                } else {
                    nextRandom.set(nextRandom.get() * 25214903917L + 11);
                    target = table.getInt((int) (nextRandom.get() >> 16) % table.length());
                    if (target == 0)
                        target = (int) nextRandom.get() % (vocab.numWords() - 1) + 1;
                    if (target == w1.getIndex())
                        continue;
                    label = 0;
                }

                double f = Nd4j.getBlasWrapper().dot(l1,syn1Neg.slice(target));
                double g;
                if (f > MAX_EXP)
                    g = (label - 1) * (useAdaGrad ?  w1.getLearningRate(target,alpha) : alpha);
                else if (f < -MAX_EXP)
                    g = (label - 0) * (useAdaGrad ?  w1.getLearningRate(target,alpha) : alpha);
                else
                    g = (label - expTable[(int)((f + MAX_EXP) * (expTable.length / MAX_EXP / 2))]) *  (useAdaGrad ?  w1.getLearningRate(target,alpha) : alpha);
                if(syn0.data().dataType() == DataBuffer.DOUBLE)
                    Nd4j.getBlasWrapper().axpy(g,neu1e,l1);
                else
                    Nd4j.getBlasWrapper().axpy((float) g,neu1e,l1);

                if(syn0.data().dataType() == DataBuffer.DOUBLE)
                    Nd4j.getBlasWrapper().axpy(g,syn1Neg,l1);
                else
                    Nd4j.getBlasWrapper().axpy((float) g,syn1Neg,l1);
            }


        avgChange /=  w1.getCodes().size();


        if(useAdaGrad) {
            if(syn0.data().dataType() == DataBuffer.DOUBLE)
                Nd4j.getBlasWrapper().axpy(avgChange,neu1e,l1);
            else
                Nd4j.getBlasWrapper().axpy((float) avgChange,neu1e,l1);


        }
        else {
            if(syn0.data().dataType() == DataBuffer.DOUBLE)
                Nd4j.getBlasWrapper().axpy(1.0,neu1e,l1);

            else
                Nd4j.getBlasWrapper().axpy(1.0f,neu1e,l1);

        }


    }
}
