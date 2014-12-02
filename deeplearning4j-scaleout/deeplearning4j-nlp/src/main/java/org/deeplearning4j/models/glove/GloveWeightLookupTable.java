package org.deeplearning4j.models.glove;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Created by agibsonccc on 12/2/14.
 */
public class GloveWeightLookupTable extends InMemoryLookupTable {


    private INDArray gradSq;
    private List<Cooccurrence> cooccurrenceList = new CopyOnWriteArrayList<>();
    private INDArray bias,gradSqBias;
    private double xMax = 0.75;

    public GloveWeightLookupTable(VocabCache vocab, int vectorLength, boolean useAdaGrad, double lr, RandomGenerator gen, double negative) {
        super(vocab, vectorLength, useAdaGrad, lr, gen, negative);
    }


    /**
     * Reset the weights of the cache
     */
    @Override
    public void resetWeights() {
        this.rng = new MersenneTwister(seed);

        //note the +2 which is the unk vocab word and the bias
        syn0  = Nd4j.rand(new int[]{vocab.numWords() + 2, vectorLength}, rng).subi(0.5).divi(vectorLength);
        gradSq =  Nd4j.ones(new int[]{vocab.numWords() + 2, vectorLength});
        INDArray randUnk = Nd4j.rand(1,vectorLength,rng).subi(0.5).divi(vectorLength);
        putVector(Word2Vec.UNK,randUnk);
        //right after unknown
        bias = syn0.slice(syn0.rows() - 1);
        gradSqBias = gradSq.slice(gradSq.rows() - 1);
        syn1 = Nd4j.create(syn0.shape());
        initNegative();

    }


    @Override
    public void iterateSample(VocabWord w1, VocabWord w2, AtomicLong nextRandom, double alpha) {
        INDArray w1Vector = syn0.slice(w1.getIndex());
        INDArray w2Vector = syn0.slice(w2.getIndex());
        Cooccurrence co = null;
        double diff = Nd4j.getBlasWrapper().dot(w1Vector,w2Vector) + (bias.getDouble(w1.getIndex()) + bias.getDouble(w2.getIndex()) - Math.log(co.getScore()));
        double fDiff = co.getScore() > xMax ? diff : Math.pow(co.getScore() / xMax,alpha) * diff;
        INDArray temp1 =  w1Vector.mul(fDiff);
        INDArray temp2 = w2Vector.mul(fDiff);

        INDArray grad1 = gradSq.slice(w1.getIndex());
        INDArray grad2 = gradSq.slice(w2.getIndex());
        grad1.addi(Transforms.pow(temp1,2));
        grad2.addi(Transforms.pow(temp2,2));

        bias.putScalar(w1.getIndex(),bias.getDouble(w1.getIndex()) - fDiff / Math.sqrt(gradSqBias.getDouble(w1.getIndex())));
        bias.putScalar(w2.getIndex(),bias.getDouble(w1.getIndex()) - fDiff / Math.sqrt(gradSqBias.getDouble(w2.getIndex())));
        fDiff *= fDiff;
        grad1.addi(fDiff);
        grad2.addi(fDiff);


    }
}
