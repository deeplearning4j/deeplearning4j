/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.models.embeddings.inmemory;

import com.google.common.util.concurrent.AtomicDouble;
import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;
import org.apache.commons.math3.util.FastMath;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.plot.BarnesHutTsne;
import org.deeplearning4j.ui.UiConnectionInfo;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.legacy.AdaGrad;
import org.nd4j.linalg.util.ArrayUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.net.HttpURLConnection;
import java.net.URI;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;


/**
 * Default word lookup table
 *
 * @author Adam Gibson
 */
public class InMemoryLookupTable<T extends SequenceElement> implements WeightLookupTable<T> {

    private static final Logger log = LoggerFactory.getLogger(InMemoryLookupTable.class);

    protected INDArray syn0, syn1;
    protected int vectorLength;
    protected transient Random rng = Nd4j.getRandom();
    protected AtomicDouble lr = new AtomicDouble(25e-3);
    protected double[] expTable;
    protected static double MAX_EXP = 6;
    protected long seed = 123;
    //negative sampling table
    protected INDArray table, syn1Neg;
    protected boolean useAdaGrad;
    protected double negative = 0;
    protected boolean useHS = true;
    protected VocabCache<T> vocab;
    protected Map<Integer, INDArray> codes = new ConcurrentHashMap<>();



    protected AdaGrad adaGrad;

    @Getter
    @Setter
    protected Long tableId;

    public InMemoryLookupTable() {}

    public InMemoryLookupTable(VocabCache<T> vocab, int vectorLength, boolean useAdaGrad, double lr, Random gen,
                    double negative, boolean useHS) {
        this(vocab, vectorLength, useAdaGrad, lr, gen, negative);
        this.useHS = useHS;
    }

    public InMemoryLookupTable(VocabCache<T> vocab, int vectorLength, boolean useAdaGrad, double lr, Random gen,
                    double negative) {
        this.vocab = vocab;
        this.vectorLength = vectorLength;
        this.useAdaGrad = useAdaGrad;
        this.lr.set(lr);
        this.rng = gen;
        this.negative = negative;
        initExpTable();

        if (useAdaGrad) {
            initAdaGrad();
        }
    }

    protected void initAdaGrad() {
        int[] shape = new int[] {vocab.numWords() + 1, vectorLength};
        int length = ArrayUtil.prod(shape);
        adaGrad = new AdaGrad(shape, lr.get());
        adaGrad.setStateViewArray(Nd4j.zeros(shape).reshape(1, length), shape, Nd4j.order(), true);

    }

    public double[] getExpTable() {
        return expTable;
    }

    public void setExpTable(double[] expTable) {
        this.expTable = expTable;
    }

    public double getGradient(int column, double gradient) {
        if (adaGrad == null)
            initAdaGrad();

        // FIXME: int cast
        return adaGrad.getGradient(gradient, column, ArrayUtil.toInts(syn0.shape()));
    }

    @Override
    public int layerSize() {
        return vectorLength;
    }

    @Override
    public void resetWeights(boolean reset) {
        if (this.rng == null)
            this.rng = Nd4j.getRandom();

        this.rng.setSeed(seed);

        if (syn0 == null || reset) {
            syn0 = Nd4j.rand(new int[] {vocab.numWords(), vectorLength}, rng).subi(0.5).divi(vectorLength);
            //            INDArray randUnk = Nd4j.rand(1, vectorLength, rng).subi(0.5).divi(vectorLength);
            //            putVector(Word2Vec.UNK, randUnk);
        }
        if ((syn1 == null || reset) && useHS) {
            log.info("Initializing syn1...");
            syn1 = Nd4j.create(syn0.shape());
        }

        initNegative();
    }

    private List<String> fitTnseAndGetLabels(final BarnesHutTsne tsne, final int numWords) {
        INDArray array = Nd4j.create(numWords, vectorLength);
        List<String> labels = new ArrayList<>();
        for (int i = 0; i < numWords && i < vocab.numWords(); i++) {
            labels.add(vocab.wordAtIndex(i));
            array.putRow(i, syn0.slice(i));
        }
        tsne.fit(array);
        return labels;
    }


    @Override
    public void plotVocab(BarnesHutTsne tsne, int numWords, File file) {
        final List<String> labels = fitTnseAndGetLabels(tsne, numWords);
        try {
            tsne.saveAsFile(labels, file.getAbsolutePath());
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Render the words via tsne
     */
    @Override
    public void plotVocab(int numWords, File file) {
        BarnesHutTsne tsne = new BarnesHutTsne.Builder().normalize(false).setFinalMomentum(0.8f).numDimension(2)
                        .setMaxIter(1000).build();
        plotVocab(tsne, numWords, file);
    }

    /**
     * Render the words via tsne
     */
    @Override
    public void plotVocab(int numWords, UiConnectionInfo connectionInfo) {
        BarnesHutTsne tsne = new BarnesHutTsne.Builder().normalize(false).setFinalMomentum(0.8f).numDimension(2)
                        .setMaxIter(1000).build();
        plotVocab(tsne, numWords, connectionInfo);
    }

    /**
     * Render the words via TSNE
     *
     * @param tsne           the tsne to use
     * @param numWords
     * @param connectionInfo
     */
    @Override
    public void plotVocab(BarnesHutTsne tsne, int numWords, UiConnectionInfo connectionInfo) {
        try {
            final List<String> labels = fitTnseAndGetLabels(tsne, numWords);
            final INDArray reducedData = tsne.getData();
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < reducedData.rows() && i < numWords; i++) {
                String word = labels.get(i);
                INDArray wordVector = reducedData.getRow(i);
                for (int j = 0; j < wordVector.length(); j++) {
                    sb.append(String.valueOf(wordVector.getDouble(j))).append(",");
                }
                sb.append(word);
            }

            String address = connectionInfo.getFirstPart() + "/tsne/post/" + connectionInfo.getSessionId();
            //            System.out.println("ADDRESS: " + address);
            URI uri = new URI(address);

            HttpURLConnection connection = (HttpURLConnection) uri.toURL().openConnection();
            connection.setRequestMethod("POST");
            connection.setRequestProperty("User-Agent", "Mozilla/5.0");
            //            connection.setRequestProperty("Content-Type", "application/json");
            connection.setRequestProperty("Content-Type", "multipart/form-data; boundary=-----TSNE-POST-DATA-----");
            connection.setDoOutput(true);

            final OutputStream outputStream = connection.getOutputStream();
            final PrintWriter writer = new PrintWriter(outputStream);
            writer.println("-------TSNE-POST-DATA-----");
            writer.println("Content-Disposition: form-data; name=\"fileupload\"; filename=\"tsne.csv\"");
            writer.println("Content-Type: text/plain; charset=UTF-16");
            writer.println("Content-Transfer-Encoding: binary");
            writer.println();
            writer.flush();

            DataOutputStream dos = new DataOutputStream(outputStream);
            dos.writeBytes(sb.toString());
            dos.flush();
            writer.println();
            writer.flush();
            dos.close();
            outputStream.close();

            try {
                int responseCode = connection.getResponseCode();
                System.out.println("RESPONSE CODE: " + responseCode);

                if (responseCode != 200) {
                    BufferedReader in = new BufferedReader(new InputStreamReader(connection.getInputStream()));
                    String inputLine;
                    StringBuilder response = new StringBuilder();

                    while ((inputLine = in.readLine()) != null) {
                        response.append(inputLine);
                    }
                    in.close();

                    log.warn("Error posting to remote UI - received response code {}\tContent: {}", response,
                                    response.toString());
                }
            } catch (IOException e) {
                log.warn("Error posting to remote UI at {}", uri, e);
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * @param codeIndex
     * @param code
     */
    @Override
    public void putCode(int codeIndex, INDArray code) {
        codes.put(codeIndex, code);
    }

    /**
     * Loads the co-occurrences for the given codes
     *
     * @param codes the codes to load
     * @return an ndarray of code.length by layerSize
     */
    @Override
    public INDArray loadCodes(int[] codes) {
        return syn1.getRows(codes);
    }


    public synchronized void initNegative() {
        if (negative > 0 && syn1Neg == null) {
            syn1Neg = Nd4j.zeros(syn0.shape());
            makeTable(Math.max(expTable.length, 100000), 0.75);
        }
    }


    protected void initExpTable() {
        expTable = new double[100000];
        for (int i = 0; i < expTable.length; i++) {
            double tmp = FastMath.exp((i / (double) expTable.length * 2 - 1) * MAX_EXP);
            expTable[i] = tmp / (tmp + 1.0);
        }
    }



    /**
     * Iterate on the given 2 vocab words
     *
     * @param w1 the first word to iterate on
     * @param w2 the second word to iterate on
     * @param nextRandom next random for sampling
     */
    @Override
    @Deprecated
    public void iterateSample(T w1, T w2, AtomicLong nextRandom, double alpha) {
        if (w2 == null || w2.getIndex() < 0 || w1.getIndex() == w2.getIndex() || w1.getLabel().equals("STOP")
                        || w2.getLabel().equals("STOP") || w1.getLabel().equals("UNK") || w2.getLabel().equals("UNK"))
            return;
        //current word vector
        INDArray l1 = this.syn0.slice(w2.getIndex());


        //error for current word and context
        INDArray neu1e = Nd4j.create(vectorLength);


        for (int i = 0; i < w1.getCodeLength(); i++) {
            int code = w1.getCodes().get(i);
            int point = w1.getPoints().get(i);
            if (point >= syn0.rows() || point < 0)
                throw new IllegalStateException("Illegal point " + point);
            //other word vector

            INDArray syn1 = this.syn1.slice(point);


            double dot = Nd4j.getBlasWrapper().dot(l1, syn1);

            if (dot < -MAX_EXP || dot >= MAX_EXP)
                continue;


            int idx = (int) ((dot + MAX_EXP) * ((double) expTable.length / MAX_EXP / 2.0));
            if (idx >= expTable.length)
                continue;

            //score
            double f = expTable[idx];
            //gradient
            double g = useAdaGrad ? w1.getGradient(i, (1 - code - f), lr.get()) : (1 - code - f) * alpha;

            Nd4j.getBlasWrapper().level1().axpy(syn1.length(), g, syn1, neu1e);
            Nd4j.getBlasWrapper().level1().axpy(syn1.length(), g, l1, syn1);

        }


        int target = w1.getIndex();
        int label;
        //negative sampling
        if (negative > 0)
            for (int d = 0; d < negative + 1; d++) {
                if (d == 0)
                    label = 1;
                else {
                    nextRandom.set(nextRandom.get() * 25214903917L + 11);

                    // FIXME: int cast
                    int idx = (int) Math.abs((int) (nextRandom.get() >> 16) % table.length());

                    target = table.getInt(idx);
                    if (target <= 0)
                        target = (int) nextRandom.get() % (vocab.numWords() - 1) + 1;

                    if (target == w1.getIndex())
                        continue;
                    label = 0;
                }


                if (target >= syn1Neg.rows() || target < 0)
                    continue;

                double f = Nd4j.getBlasWrapper().dot(l1, syn1Neg.slice(target));
                double g;
                if (f > MAX_EXP)
                    g = useAdaGrad ? w1.getGradient(target, (label - 1), alpha) : (label - 1) * alpha;
                else if (f < -MAX_EXP)
                    g = label * (useAdaGrad ? w1.getGradient(target, alpha, alpha) : alpha);
                else
                    g = useAdaGrad ? w1
                                    .getGradient(target,
                                                    label - expTable[(int) ((f + MAX_EXP)
                                                                    * (expTable.length / MAX_EXP / 2))],
                                                    alpha)
                                    : (label - expTable[(int) ((f + MAX_EXP) * (expTable.length / MAX_EXP / 2))])
                                                    * alpha;
                if (syn0.data().dataType() == DataBuffer.Type.DOUBLE)
                    Nd4j.getBlasWrapper().axpy(g, syn1Neg.slice(target), neu1e);
                else
                    Nd4j.getBlasWrapper().axpy((float) g, syn1Neg.slice(target), neu1e);

                if (syn0.data().dataType() == DataBuffer.Type.DOUBLE)
                    Nd4j.getBlasWrapper().axpy(g, l1, syn1Neg.slice(target));
                else
                    Nd4j.getBlasWrapper().axpy((float) g, l1, syn1Neg.slice(target));
            }

        if (syn0.data().dataType() == DataBuffer.Type.DOUBLE)
            Nd4j.getBlasWrapper().axpy(1.0, neu1e, l1);

        else
            Nd4j.getBlasWrapper().axpy(1.0f, neu1e, l1);

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

    public void setUseHS(boolean useHS) {
        this.useHS = useHS;
    }

    public void setNegative(double negative) {
        this.negative = negative;
    }

    /**
     * Iterate on the given 2 vocab words
     *
     * @param w1 the first word to iterate on
     * @param w2 the second word to iterate on
     */
    @Deprecated
    @Override
    public void iterate(T w1, T w2) {

    }


    /**
     * Reset the weights of the cache
     */
    @Override
    public void resetWeights() {
        resetWeights(true);
    }


    protected void makeTable(int tableSize, double power) {
        int vocabSize = syn0.rows();
        table = Nd4j.create(tableSize);
        double trainWordsPow = 0.0;
        for (String word : vocab.words()) {
            trainWordsPow += Math.pow(vocab.wordFrequency(word), power);
        }

        int wordIdx = 0;
        String word = vocab.wordAtIndex(wordIdx);
        double d1 = Math.pow(vocab.wordFrequency(word), power) / trainWordsPow;
        for (int i = 0; i < tableSize; i++) {
            table.putScalar(i, wordIdx);
            double mul = i * 1.0 / (double) tableSize;
            if (mul > d1) {
                if (wordIdx < vocabSize - 1)
                    wordIdx++;
                word = vocab.wordAtIndex(wordIdx);
                String wordAtIndex = vocab.wordAtIndex(wordIdx);
                if (word == null)
                    continue;
                d1 += Math.pow(vocab.wordFrequency(wordAtIndex), power) / trainWordsPow;
            }
        }
    }

    /**
     * Inserts a word vector
     *
     * @param word   the word to insert
     * @param vector the vector to insert
     */
    @Override
    public void putVector(String word, INDArray vector) {
        if (word == null)
            throw new IllegalArgumentException("No null words allowed");
        if (vector == null)
            throw new IllegalArgumentException("No null vectors allowed");
        int idx = vocab.indexOf(word);
        syn0.slice(idx).assign(vector);

    }

    public INDArray getTable() {
        return table;
    }

    public void setTable(INDArray table) {
        this.table = table;
    }

    public INDArray getSyn1Neg() {
        return syn1Neg;
    }

    public void setSyn1Neg(INDArray syn1Neg) {
        this.syn1Neg = syn1Neg;
    }

    /**
     * @param word
     * @return
     */
    @Override
    public INDArray vector(String word) {
        if (word == null)
            return null;
        int idx = vocab.indexOf(word);
        if (idx < 0) {
            idx = vocab.indexOf(Word2Vec.DEFAULT_UNK);
            if (idx < 0)
                return null;
        }
        return syn0.getRow(idx);
    }

    @Override
    public void setLearningRate(double lr) {
        this.lr.set(lr);
    }

    @Override
    public Iterator<INDArray> vectors() {
        return new WeightIterator();
    }

    @Override
    public INDArray getWeights() {
        return syn0;
    }


    protected class WeightIterator implements Iterator<INDArray> {
        protected int currIndex = 0;

        @Override
        public boolean hasNext() {
            return currIndex < syn0.rows();
        }

        @Override
        public INDArray next() {
            INDArray ret = syn0.slice(currIndex);
            currIndex++;
            return ret;
        }

        @Override
        public void remove() {
            throw new UnsupportedOperationException();
        }
    }

    public INDArray getSyn0() {
        return syn0;
    }

    public void setSyn0(INDArray syn0) {
        this.syn0 = syn0;
    }

    public INDArray getSyn1() {
        return syn1;
    }

    public void setSyn1(INDArray syn1) {
        this.syn1 = syn1;
    }

    @Override
    public VocabCache<T> getVocabCache() {
        return vocab;
    }

    public void setVectorLength(int vectorLength) {
        this.vectorLength = vectorLength;
    }

    /**
     * This method is deprecated, since all logic was pulled out from this class and is not used anymore.
     * However this method will be around for a while, due to backward compatibility issues.
     * @return initial learning rate
     */
    @Deprecated
    public AtomicDouble getLr() {
        return lr;
    }

    public void setLr(AtomicDouble lr) {
        this.lr = lr;
    }

    public VocabCache getVocab() {
        return vocab;
    }

    public void setVocab(VocabCache vocab) {
        this.vocab = vocab;
    }

    public Map<Integer, INDArray> getCodes() {
        return codes;
    }

    public void setCodes(Map<Integer, INDArray> codes) {
        this.codes = codes;
    }

    public static class Builder<T extends SequenceElement> {
        protected int vectorLength = 100;
        protected boolean useAdaGrad = false;
        protected double lr = 0.025;
        protected Random gen = Nd4j.getRandom();
        protected long seed = 123;
        protected double negative = 0;
        protected VocabCache<T> vocabCache;
        protected boolean useHS = true;



        public Builder<T> useHierarchicSoftmax(boolean reallyUse) {
            this.useHS = reallyUse;
            return this;
        }

        public Builder<T> cache(@NonNull VocabCache<T> vocab) {
            this.vocabCache = vocab;
            return this;
        }

        public Builder<T> negative(double negative) {
            this.negative = negative;
            return this;
        }

        public Builder<T> vectorLength(int vectorLength) {
            this.vectorLength = vectorLength;
            return this;
        }

        public Builder<T> useAdaGrad(boolean useAdaGrad) {
            this.useAdaGrad = useAdaGrad;
            return this;
        }

        /**
         * This method is deprecated, since all logic was pulled out from this class
         * @param lr
         * @return
         */
        @Deprecated
        public Builder<T> lr(double lr) {
            this.lr = lr;
            return this;
        }

        public Builder<T> gen(Random gen) {
            this.gen = gen;
            return this;
        }

        public Builder<T> seed(long seed) {
            this.seed = seed;
            return this;
        }



        public InMemoryLookupTable<T> build() {
            if (vocabCache == null)
                throw new IllegalStateException("Vocab cache must be specified");

            InMemoryLookupTable<T> table =
                            new InMemoryLookupTable<>(vocabCache, vectorLength, useAdaGrad, lr, gen, negative, useHS);
            table.seed = seed;

            return table;
        }
    }

    @Override
    public String toString() {
        return "InMemoryLookupTable{" + "syn0=" + syn0 + ", syn1=" + syn1 + ", vectorLength=" + vectorLength + ", rng="
                        + rng + ", lr=" + lr + ", expTable=" + Arrays.toString(expTable) + ", seed=" + seed + ", table="
                        + table + ", syn1Neg=" + syn1Neg + ", useAdaGrad=" + useAdaGrad + ", negative=" + negative
                        + ", vocab=" + vocab + ", codes=" + codes + '}';
    }

    /**
     * This method consumes weights of a given InMemoryLookupTable
     *
     * PLEASE NOTE: this method explicitly resets current weights
     *
     * @param srcTable
     */
    public void consume(InMemoryLookupTable<T> srcTable) {
        if (srcTable.vectorLength != this.vectorLength)
            throw new IllegalStateException("You can't consume lookupTable with different vector lengths");

        if (srcTable.syn0 == null)
            throw new IllegalStateException("Source lookupTable Syn0 is NULL");

        this.resetWeights(true);

        AtomicInteger cntHs = new AtomicInteger(0);
        AtomicInteger cntNg = new AtomicInteger(0);

        if (srcTable.syn0.rows() > this.syn0.rows())
            throw new IllegalStateException(
                            "You can't consume lookupTable with built for larger vocabulary without updating your vocabulary first");

        for (int x = 0; x < srcTable.syn0.rows(); x++) {
            this.syn0.putRow(x, srcTable.syn0.getRow(x));

            if (this.syn1 != null && srcTable.syn1 != null)
                this.syn1.putRow(x, srcTable.syn1.getRow(x));
            else if (cntHs.incrementAndGet() == 1)
                log.info("Skipping syn1 merge");

            if (this.syn1Neg != null && srcTable.syn1Neg != null) {
                this.syn1Neg.putRow(x, srcTable.syn1Neg.getRow(x));
            } else if (cntNg.incrementAndGet() == 1)
                log.info("Skipping syn1Neg merge");

            if (cntHs.get() > 0 && cntNg.get() > 0)
                throw new ND4JIllegalStateException("srcTable has no syn1/syn1neg");
        }
    }
}
