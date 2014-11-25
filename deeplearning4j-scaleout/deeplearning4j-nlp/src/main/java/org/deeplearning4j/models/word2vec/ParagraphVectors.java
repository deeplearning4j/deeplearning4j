package org.deeplearning4j.models.word2vec;

import com.google.common.base.Function;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.text.tokenization.tokenizerfactory.UimaTokenizerFactory;

import javax.annotation.Nullable;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Known in some circles as doc2vec. Officially called paragraph vectors.
 * See:
 * http://arxiv.org/pdf/1405.4053v2.pdf
 *
 * @author Adam Gibson
 */
public class ParagraphVectors extends Word2Vec {


    private boolean trainLabels = true;
    private boolean trainWords = true;

    public void dm(List<VocabWord> sentence,boolean trainLabels,boolean trainWords) {

    }


    @Override
    public void fit() throws IOException {

    }

    public static class Builder extends Word2Vec.Builder {

       private boolean trainWords = true;
       private boolean trainLabels = true;


        public Builder trainWords(boolean trainWords) {
            this.trainWords = trainWords;
            return this;
        }

        public Builder trainLabels(boolean trainLabels) {
            this.trainLabels = trainLabels;
            return this;
        }

        @Override
        public ParagraphVectors build() {

            if(iter == null) {
                ParagraphVectors ret = new ParagraphVectors();
                ret.layerSize = layerSize;
                ret.window = window;
                ret.alpha.set(lr);
                ret.vectorizer = textVectorizer;
                ret.stopWords = stopWords;
                ret.setCache(vocabCache);
                ret.numIterations = iterations;
                ret.minWordFrequency = minWordFrequency;
                ret.seed = seed;
                ret.saveVocab = saveVocab;
                ret.batchSize = batchSize;
                ret.useAdaGrad = useAdaGrad;
                ret.minLearningRate = minLearningRate;
                ret.sample = sampling;
                ret.trainLabels = trainLabels;
                ret.trainWords = trainWords;

                try {
                    if (tokenizerFactory == null)
                        tokenizerFactory = new UimaTokenizerFactory();
                }catch(Exception e) {
                    throw new RuntimeException(e);
                }

                if(vocabCache == null) {
                    vocabCache = new InMemoryLookupCache.Builder().negative(negative)
                            .useAdaGrad(useAdaGrad).lr(lr)
                            .vectorLength(layerSize).build();

                    ret.cache = vocabCache;
                }
                ret.docIter = docIter;
                ret.tokenizerFactory = tokenizerFactory;

                return ret;
            }

            else {
                ParagraphVectors ret = new ParagraphVectors();
                ret.alpha.set(lr);
                ret.layerSize = layerSize;
                ret.sentenceIter = iter;
                ret.window = window;
                ret.useAdaGrad = useAdaGrad;
                ret.minLearningRate = minLearningRate;
                ret.vectorizer = textVectorizer;
                ret.stopWords = stopWords;
                ret.minWordFrequency = minWordFrequency;
                ret.setCache(vocabCache);
                ret.docIter = docIter;
                ret.minWordFrequency = minWordFrequency;
                ret.numIterations = iterations;
                ret.seed = seed;
                ret.numIterations = iterations;
                ret.saveVocab = saveVocab;
                ret.batchSize = batchSize;
                ret.sample = sampling;
                ret.trainLabels = trainLabels;
                ret.trainWords = trainWords;

                try {
                    if (tokenizerFactory == null)
                        tokenizerFactory = new UimaTokenizerFactory();
                }catch(Exception e) {
                    throw new RuntimeException(e);
                }

                if(vocabCache == null) {
                    vocabCache = new InMemoryLookupCache.Builder().negative(negative)
                            .useAdaGrad(useAdaGrad).lr(lr)
                            .vectorLength(layerSize).build();
                    ret.cache = vocabCache;
                }
                ret.tokenizerFactory = tokenizerFactory;
                return ret;
            }



        }
    }

}
