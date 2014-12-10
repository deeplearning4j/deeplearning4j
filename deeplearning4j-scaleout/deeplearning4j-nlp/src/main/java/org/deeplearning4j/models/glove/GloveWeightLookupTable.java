package org.deeplearning4j.models.glove;



import org.apache.commons.io.IOUtils;
import org.apache.commons.io.LineIterator;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.AdaGrad;


import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Glove lookup table
 *
 * @author Adam Gibson
 */
public class GloveWeightLookupTable extends InMemoryLookupTable {


    private AdaGrad weightAdaGrad;
    private AdaGrad biasAdaGrad;
    private INDArray bias;
    //also known as alpha
    private double xMax = 0.75;
    private double maxCount = 100;


    public GloveWeightLookupTable(VocabCache vocab, int vectorLength, boolean useAdaGrad, double lr, RandomGenerator gen, double negative, double xMax,double maxCount) {
        super(vocab, vectorLength, useAdaGrad, lr, gen, negative);
        this.xMax = xMax;
        this.maxCount = maxCount;
    }

    @Override
    public void resetWeights(boolean reset) {
        if(rng == null)
            this.rng = new MersenneTwister(seed);

        //note the +2 which is the unk vocab word and the bias
        if(syn0 == null || syn0 != null && reset) {
            syn0 = Nd4j.rand(new int[]{vocab.numWords() + 1, vectorLength}, rng).subi(0.5).divi((double) vectorLength);
            INDArray randUnk = Nd4j.rand(1,vectorLength,rng).subi(0.5).divi(vectorLength);
            putVector(Word2Vec.UNK, randUnk);
        }
        if(weightAdaGrad == null || weightAdaGrad != null && reset) {
            weightAdaGrad = new AdaGrad(new int[]{vocab.numWords() + 1, vectorLength});
            weightAdaGrad.setMasterStepSize(lr.get());
        }


        //right after unknown
        if(bias == null || bias != null && reset)
            bias = Nd4j.create(syn0.rows());

        if(biasAdaGrad == null || biasAdaGrad != null && reset) {
            biasAdaGrad = new AdaGrad(bias.shape());
            biasAdaGrad.setMasterStepSize(lr.get());
        }


    }

    /**
     * Reset the weights of the cache
     */
    @Override
    public void resetWeights() {
        if(rng == null)
            this.rng = new MersenneTwister(seed);

        //note the +2 which is the unk vocab word and the bias
        syn0  = Nd4j.rand(new int[]{vocab.numWords() + 1, vectorLength}, rng).subi(0.5).divi((double) vectorLength);
        weightAdaGrad = new AdaGrad(new int[]{vocab.numWords() + 1, vectorLength});
        weightAdaGrad.setMasterStepSize(lr.get());
        INDArray randUnk = Nd4j.rand(1,vectorLength,rng).subi(0.5).divi(vectorLength);
        putVector(Word2Vec.UNK,randUnk);
        //right after unknown
        bias = Nd4j.create(syn0.rows());
        biasAdaGrad = new AdaGrad(bias.shape());
        biasAdaGrad.setMasterStepSize(lr.get());

    }

    /**
     * glove iteration
     * @param w1 the first word
     * @param w2 the second word
     * @param score the weight learned for the particular co occurrences
     */
    public   double iterateSample(VocabWord w1, VocabWord w2,double score) {
        INDArray w1Vector = syn0.slice(w1.getIndex());
        INDArray w2Vector = syn0.slice(w2.getIndex());
        //prediction: input + bias
        if(w1.getIndex() < 0 || w1.getIndex() >= syn0.rows())
            throw new IllegalArgumentException("Illegal index for word " + w1.getWord());
        if(w2.getIndex() < 0 || w2.getIndex() >= syn0.rows())
            throw new IllegalArgumentException("Illegal index for word " + w2.getWord());


        //w1 * w2 + bias
        double prediction = Nd4j.getBlasWrapper().dot(w1Vector,w2Vector);
        prediction +=  bias.getDouble(w1.getIndex()) + bias.getDouble(w2.getIndex());

        double weight = Math.pow(Math.min(1.0,(score / maxCount)),xMax);

        double fDiff = weight * (prediction - Math.log(score));


        //amount of change
        double gradient =  fDiff;

        //note the update step here: the gradient is
        //the gradient of the OPPOSITE word
        //for adagrad we will use the index of the word passed in
        //for the gradient calculation we will use the context vector
        update(w1,w1Vector,w2Vector.dup(),gradient);
        update(w2,w2Vector,w1Vector.dup(),gradient);
        return fDiff;



    }


    private void update(VocabWord w1,INDArray wordVector,INDArray contextVector,double gradient) {
        //gradient for word vectors
        INDArray grad1 =  contextVector.mul(gradient);
        INDArray update = weightAdaGrad.getGradient(grad1,w1.getIndex(),syn0.shape());

        //update vector
        wordVector.subi(update);

        double w1Bias = bias.getDouble(w1.getIndex());
        double biasGradient = biasAdaGrad.getGradient(gradient,w1.getIndex(),bias.shape());
        double update2 = w1Bias - biasGradient;
        bias.putScalar(w1.getIndex(),update2);
    }

    public AdaGrad getWeightAdaGrad() {
        return weightAdaGrad;
    }

    public void setWeightAdaGrad(AdaGrad weightAdaGrad) {
        this.weightAdaGrad = weightAdaGrad;
    }

    public AdaGrad getBiasAdaGrad() {
        return biasAdaGrad;
    }

    public void setBiasAdaGrad(AdaGrad biasAdaGrad) {
        this.biasAdaGrad = biasAdaGrad;
    }

    /**
     * Load the weights in raw format
     * @param is the input stream to read from for the weights
     * @param vocab the vocab for the lookuptable
     * @param vectorLength the length of each row
     * @return the lookup table
     * @throws IOException
     */
    public static GloveWeightLookupTable loadRawArray(InputStream is,VocabCache vocab,int vectorLength) throws IOException {
       GloveWeightLookupTable ret = new GloveWeightLookupTable.Builder()
               .cache(vocab).vectorLength(vectorLength)
               .build();
        INDArray syn0 = Nd4j.readTxt(is," ");
        ret.setSyn0(syn0);
        ret.resetWeights(false);
        return ret;

    }

    /**
     * Load a glove model from an input stream.
     * The format is:
     * word num1 num2....
     * @param is the input stream to read from for the weights
     * @param vocab the vocab for the lookuptable
     * @return the loaded model
     * @throws java.io.IOException if one occurs
     */
    public static GloveWeightLookupTable load(InputStream is,VocabCache vocab) throws IOException {
        LineIterator iter = IOUtils.lineIterator(is, "UTF-8");
        GloveWeightLookupTable glove = null;
        Map<String,float[]> wordVectors = new HashMap<>();
        while(iter.hasNext()) {
            String line = iter.nextLine().trim();
            if(line.isEmpty())
                continue;
            String[] split = line.split(" ");
            String word = split[0];
            if(glove == null)
                glove = new GloveWeightLookupTable.Builder()
                        .cache(vocab).vectorLength(split.length - 1)
                        .build();



            if(word.isEmpty())
                continue;
            float[] read = read(split,glove.getVectorLength());
            if(read.length < 1)
                continue;


            wordVectors.put(word,read);



        }

        glove.setSyn0(weights(glove,wordVectors,vocab));
        glove.resetWeights(false);


        iter.close();


        return glove;

    }

    private static INDArray weights(GloveWeightLookupTable glove,Map<String,float[]> data,VocabCache vocab) {
        INDArray ret = Nd4j.create(data.size(),glove.getVectorLength());
        for(String key : data.keySet()) {
            INDArray row = Nd4j.create(Nd4j.createBuffer(data.get(key)));
            if(row.length() != glove.getVectorLength())
                continue;
            if(vocab.indexOf(key) >= data.size())
                continue;
            if(vocab.indexOf(key) < 0)
                continue;
            ret.putRow(vocab.indexOf(key), row);
        }
        return ret;
    }


    private static float[] read(String[] split,int length) {
        float[] ret = new float[length];
        for(int i = 1; i < split.length; i++) {
            ret[i - 1] = Float.parseFloat(split[i]);
        }
        return ret;
    }


    @Override
    public void iterateSample(VocabWord w1, VocabWord w2, AtomicLong nextRandom, double alpha) {
        throw new UnsupportedOperationException();

    }

    public INDArray getBias() {
        return bias;
    }

    public void setBias(INDArray bias) {
        this.bias = bias;
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
