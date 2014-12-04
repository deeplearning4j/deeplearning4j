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

import java.util.concurrent.atomic.AtomicLong;

/**
 * Glove lookup table
 *
 * @author Adam Gibson
 */
public class GloveWeightLookupTable extends InMemoryLookupTable {


    private INDArray gradSq;
    private INDArray bias,gradSqBias;
    //also known as alpha
    private double xMax = 0.75;
    private double maxCount = 100;


    public GloveWeightLookupTable(VocabCache vocab, int vectorLength, boolean useAdaGrad, double lr, RandomGenerator gen, double negative, double xMax,double maxCount) {
        super(vocab, vectorLength, useAdaGrad, lr, gen, negative);
        this.xMax = xMax;
        this.maxCount = maxCount;
    }

    /**
     * Reset the weights of the cache
     */
    @Override
    public void resetWeights() {
        this.rng = new MersenneTwister(seed);

        //note the +2 which is the unk vocab word and the bias
        syn0  = Nd4j.rand(new int[]{vocab.numWords() + 1, vectorLength}, rng).subi(0.5).divi(vectorLength);
        gradSq =  Nd4j.ones(new int[]{vocab.numWords() + 1, vectorLength});
        INDArray randUnk = Nd4j.rand(1,vectorLength,rng).subi(0.5).divi(vectorLength);
        putVector(Word2Vec.UNK,randUnk);
        //right after unknown
        bias = Nd4j.create(syn0.rows());
        gradSqBias = Nd4j.create(bias.shape());
        initNegative();

    }

    /**
     * glove iteration
     * @param w1 the first word
     * @param w2 the second word
     * @param score the weight learned for the particular co occurrences
     */
    public void iterateSample(VocabWord w1, VocabWord w2,double score) {
        INDArray w1Vector = syn0.slice(w1.getIndex());
        INDArray w2Vector = syn0.slice(w2.getIndex());
        //prediction: input + bias
       if(w1.getIndex() < 0 || w1.getIndex() >= syn0.rows())
           throw new IllegalArgumentException("Illegal index for word " + w1.getWord());
        if(w2.getIndex() < 0 || w2.getIndex() >= syn0.rows())
            throw new IllegalArgumentException("Illegal index for word " + w2.getWord());

        double bias2 = this.bias.getDouble(w1.getIndex()) + this.bias.getDouble(w2.getIndex());
        double prediction = Nd4j.getBlasWrapper().dot(w1Vector,w2Vector) + bias2;
        double weight = Math.pow(Math.min(1.0,(score / maxCount)),xMax);
        double loss =   weight * (prediction - Math.log(score));


        //adagrad sum squared gradients
        INDArray gradSq1 = gradSq.slice(w1.getIndex());
        INDArray gradSq2 = gradSq.slice(w2.getIndex());


        //adagrad learning rates based on the history
        INDArray grad1LearningRates = Transforms.sqrt(gradSq1).rdivi(lr);
        INDArray grad2LearningRates = Transforms.sqrt(gradSq2).rdivi(lr);


        //gradient for word vectors
        INDArray grad1 =  w1Vector.mul(loss);
        INDArray grad2 = w2Vector.mul(loss);

        //update vector
        w1Vector.subi(grad1.mul(grad1LearningRates));
        w2Vector.subi(grad2.mul(grad2LearningRates));

        //update the squared gradient history
        gradSq1.addi(Transforms.pow(grad1, 2));
        gradSq2.addi(Transforms.pow(grad2, 2));



        //update biases
        double w1SqBias = gradSqBias.getDouble(w1.getIndex());
        double w2SqBias = gradSqBias.getDouble(w2.getIndex());

        double w1Lr = lr.get() / Math.sqrt(w1SqBias);
        double w2Lr = lr.get() / Math.sqrt(w2SqBias);

        double w1Bias = bias.getDouble(w1.getIndex());
        double w2Bias = bias.getDouble(w2.getIndex());

        double update = w1Bias - (loss * w1Lr);
        double update2 = w2Bias - (loss * w2Lr);

        w1SqBias += Math.pow(loss,2);
        w2SqBias += Math.pow(loss,2);


        bias.putScalar(w1.getIndex(),update);
        bias.putScalar(w2.getIndex(),update2);

        gradSqBias.putScalar(w1.getIndex(),w1SqBias);
        gradSqBias.putScalar(w2.getIndex(),w2SqBias);




    }


    @Override
    public void iterateSample(VocabWord w1, VocabWord w2, AtomicLong nextRandom, double alpha) {
        throw new UnsupportedOperationException();

    }


    public static class Builder extends  InMemoryLookupTable.Builder {
        private double xMax = 0.75;
        private double maxCount = 100;


        public Builder maxCount(double maxCount) {
            this.maxCount = maxCount;
            return this;
        }


        public Builder xMax(double xMax) {
            this.xMax = xMax;
            return this;
        }

        @Override
        public Builder cache(VocabCache vocab) {
            super.cache(vocab);
            return this;
        }

        @Override
        public Builder negative(double negative) {
            super.negative(negative);
            return this;
        }

        @Override
        public Builder vectorLength(int vectorLength) {
            super.vectorLength(vectorLength);
            return this;
        }

        @Override
        public Builder useAdaGrad(boolean useAdaGrad) {
            super.useAdaGrad(useAdaGrad);
            return this;
        }

        @Override
        public Builder lr(double lr) {
            super.lr(lr);
            return this;
        }

        @Override
        public Builder gen(RandomGenerator gen) {
            super.gen(gen);
            return this;
        }

        @Override
        public Builder seed(long seed) {
            super.seed(seed);
            return this;
        }

        public GloveWeightLookupTable build() {
            return new GloveWeightLookupTable(vocabCache,vectorLength,useAdaGrad,lr,gen,negative,xMax,maxCount);
        }
    }

}
