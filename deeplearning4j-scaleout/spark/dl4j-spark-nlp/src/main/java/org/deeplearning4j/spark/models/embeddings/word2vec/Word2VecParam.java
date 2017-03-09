/*-
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.spark.models.embeddings.word2vec;

import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Serializable;
import java.util.concurrent.atomic.AtomicLong;

/**
 * @author Adam Gibson
 */
@Deprecated
public class Word2VecParam implements Serializable {

    private boolean useAdaGrad = false;
    private double negative = 5;
    private int numWords = 1;
    private INDArray table;
    private int window = 5;
    private AtomicLong nextRandom = new AtomicLong(5);
    private double alpha = 0.025;
    private double minAlpha = 1e-2;
    private int totalWords = 1;
    private static transient final Logger log = LoggerFactory.getLogger(Word2VecPerformer.class);
    private int lastChecked = 0;
    private Broadcast<AtomicLong> wordCount;
    private InMemoryLookupTable weights;
    private int vectorLength;
    private Broadcast<double[]> expTable;
    private AtomicLong wordsSeen = new AtomicLong(0);
    private AtomicLong lastWords = new AtomicLong(0);

    public Word2VecParam(boolean useAdaGrad, double negative, int numWords, INDArray table, int window,
                    AtomicLong nextRandom, double alpha, double minAlpha, int totalWords, int lastChecked,
                    Broadcast<AtomicLong> wordCount, InMemoryLookupTable weights, int vectorLength,
                    Broadcast<double[]> expTable) {
        this.useAdaGrad = useAdaGrad;
        this.negative = negative;
        this.numWords = numWords;
        this.table = table;
        this.window = window;
        this.nextRandom = nextRandom;
        this.alpha = alpha;
        this.minAlpha = minAlpha;
        this.totalWords = totalWords;
        this.lastChecked = lastChecked;
        this.wordCount = wordCount;
        this.weights = weights;
        this.vectorLength = vectorLength;
        this.expTable = expTable;
    }

    public AtomicLong getLastWords() {
        return lastWords;
    }

    public void setLastWords(AtomicLong lastWords) {
        this.lastWords = lastWords;
    }

    public AtomicLong getWordsSeen() {
        return wordsSeen;
    }

    public void setWordsSeen(AtomicLong wordsSeen) {
        this.wordsSeen = wordsSeen;
    }

    public Broadcast<double[]> getExpTable() {
        return expTable;
    }

    public void setExpTable(Broadcast<double[]> expTable) {
        this.expTable = expTable;
    }

    public boolean isUseAdaGrad() {
        return useAdaGrad;
    }

    public void setUseAdaGrad(boolean useAdaGrad) {
        this.useAdaGrad = useAdaGrad;
    }

    public double getNegative() {
        return negative;
    }

    public void setNegative(double negative) {
        this.negative = negative;
    }

    public int getNumWords() {
        return numWords;
    }

    public void setNumWords(int numWords) {
        this.numWords = numWords;
    }

    public INDArray getTable() {
        return table;
    }

    public void setTable(INDArray table) {
        this.table = table;
    }

    public int getWindow() {
        return window;
    }

    public void setWindow(int window) {
        this.window = window;
    }

    public AtomicLong getNextRandom() {
        return nextRandom;
    }

    public void setNextRandom(AtomicLong nextRandom) {
        this.nextRandom = nextRandom;
    }

    public double getAlpha() {
        return alpha;
    }

    public void setAlpha(double alpha) {
        this.alpha = alpha;
    }

    public double getMinAlpha() {
        return minAlpha;
    }

    public void setMinAlpha(double minAlpha) {
        this.minAlpha = minAlpha;
    }

    public int getTotalWords() {
        return totalWords;
    }

    public void setTotalWords(int totalWords) {
        this.totalWords = totalWords;
    }

    public static Logger getLog() {
        return log;
    }

    public int getLastChecked() {
        return lastChecked;
    }

    public void setLastChecked(int lastChecked) {
        this.lastChecked = lastChecked;
    }

    public Broadcast<AtomicLong> getWordCount() {
        return wordCount;
    }

    public void setWordCount(Broadcast<AtomicLong> wordCount) {
        this.wordCount = wordCount;
    }

    public InMemoryLookupTable getWeights() {
        return weights;
    }

    public void setWeights(InMemoryLookupTable weights) {
        this.weights = weights;
    }



    public int getVectorLength() {
        return vectorLength;
    }

    public void setVectorLength(int vectorLength) {
        this.vectorLength = vectorLength;
    }

    public static class Builder {
        private boolean useAdaGrad = true;
        private double negative = 0;
        private int numWords = 1;
        private INDArray table;
        private int window = 5;
        private AtomicLong nextRandom;
        private double alpha = 0.025;
        private double minAlpha = 0.01;
        private int totalWords;
        private int lastChecked;
        private Broadcast<AtomicLong> wordCount;
        private InMemoryLookupTable weights;
        private int vectorLength = 300;
        private Broadcast<double[]> expTable;

        public Builder expTable(Broadcast<double[]> expTable) {
            this.expTable = expTable;
            return this;
        }


        public Builder useAdaGrad(boolean useAdaGrad) {
            this.useAdaGrad = useAdaGrad;
            return this;
        }

        public Builder negative(double negative) {
            this.negative = negative;
            return this;
        }

        public Builder numWords(int numWords) {
            this.numWords = numWords;
            return this;
        }

        public Builder table(INDArray table) {
            this.table = table;
            return this;
        }

        public Builder window(int window) {
            this.window = window;
            return this;
        }

        public Builder setNextRandom(AtomicLong nextRandom) {
            this.nextRandom = nextRandom;
            return this;
        }

        public Builder setAlpha(double alpha) {
            this.alpha = alpha;
            return this;
        }

        public Builder setMinAlpha(double minAlpha) {
            this.minAlpha = minAlpha;
            return this;
        }

        public Builder totalWords(int totalWords) {
            this.totalWords = totalWords;
            return this;
        }

        public Builder lastChecked(int lastChecked) {
            this.lastChecked = lastChecked;
            return this;
        }

        public Builder wordCount(Broadcast<AtomicLong> wordCount) {
            this.wordCount = wordCount;
            return this;
        }

        public Builder weights(InMemoryLookupTable weights) {
            this.weights = weights;
            return this;
        }

        public Builder setVectorLength(int vectorLength) {
            this.vectorLength = vectorLength;
            return this;
        }

        public Word2VecParam build() {
            return new Word2VecParam(useAdaGrad, negative, numWords, table, window, nextRandom, alpha, minAlpha,
                            totalWords, lastChecked, wordCount, weights, vectorLength, expTable);
        }
    }
}
