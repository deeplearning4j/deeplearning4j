/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.iterator;

import lombok.Getter;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.iterator.bert.BertMaskedLMMasker;
import org.deeplearning4j.iterator.provider.CollectionLabeledPairSentenceProvider;
import org.deeplearning4j.iterator.provider.CollectionLabeledSentenceProvider;
import org.deeplearning4j.text.tokenization.tokenizerfactory.BertWordPieceTokenizerFactory;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.common.primitives.Pair;
import org.nd4j.common.primitives.Triple;
import org.nd4j.common.resources.Resources;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.util.*;

import static org.junit.Assert.*;


public class TestBertIterator extends BaseDL4JTest {

    private static File pathToVocab = Resources.asFile("other/vocab.txt");
    private static Charset c = StandardCharsets.UTF_8;
    private static String shortSentence = "I saw a girl with a telescope.";
    private static String longSentence = "Donaudampfschifffahrts Kapitänsmützeninnenfuttersaum";
    private static String sentenceA = "Goodnight noises everywhere";
    private static String sentenceB = "Goodnight moon";

    public TestBertIterator() throws IOException {
    }

    @Test(timeout = 20000L)
    public void testBertSequenceClassification() throws Exception {

        int minibatchSize = 2;
        TestSentenceHelper testHelper = new TestSentenceHelper();
        BertIterator b = BertIterator.builder()
                .tokenizer(testHelper.getTokenizer())
                .lengthHandling(BertIterator.LengthHandling.FIXED_LENGTH, 16)
                .minibatchSize(minibatchSize)
                .sentenceProvider(testHelper.getSentenceProvider())
                .featureArrays(BertIterator.FeatureArrays.INDICES_MASK)
                .vocabMap(testHelper.getTokenizer().getVocab())
                .task(BertIterator.Task.SEQ_CLASSIFICATION)
                .build();

        MultiDataSet mds = b.next();
        assertEquals(1, mds.getFeatures().length);
        System.out.println(mds.getFeatures(0));
        System.out.println(mds.getFeaturesMaskArray(0));

        INDArray expF = Nd4j.create(DataType.INT, 1, 16);
        INDArray expM = Nd4j.create(DataType.INT, 1, 16);
        Map<String, Integer> m = testHelper.getTokenizer().getVocab();
        for (int i = 0; i < minibatchSize; i++) {
            INDArray expFTemp = Nd4j.create(DataType.INT, 1, 16);
            INDArray expMTemp = Nd4j.create(DataType.INT, 1, 16);
            List<String> tokens = testHelper.getTokenizedSentences().get(i);
            System.out.println(tokens);
            for (int j = 0; j < tokens.size(); j++) {
                String token = tokens.get(j);
                if (!m.containsKey(token)) {
                    throw new IllegalStateException("Unknown token: \"" + token + "\"");
                }
                int idx = m.get(token);
                expFTemp.putScalar(0, j, idx);
                expMTemp.putScalar(0, j, 1);
            }
            if (i == 0) {
                expF = expFTemp.dup();
                expM = expMTemp.dup();
            } else {
                expF = Nd4j.vstack(expF, expFTemp);
                expM = Nd4j.vstack(expM, expMTemp);
            }
        }
        assertEquals(expF, mds.getFeatures(0));
        assertEquals(expM, mds.getFeaturesMaskArray(0));
        assertEquals(expF, b.featurizeSentences(testHelper.getSentences()).getFirst()[0]);
        assertEquals(expM, b.featurizeSentences(testHelper.getSentences()).getSecond()[0]);

        assertFalse(b.hasNext());
        b.reset();
        assertTrue(b.hasNext());

        //Same thing, but with segment ID also
        b = BertIterator.builder()
                .tokenizer(testHelper.getTokenizer())
                .lengthHandling(BertIterator.LengthHandling.FIXED_LENGTH, 16)
                .minibatchSize(minibatchSize)
                .sentenceProvider(testHelper.getSentenceProvider())
                .featureArrays(BertIterator.FeatureArrays.INDICES_MASK_SEGMENTID)
                .vocabMap(testHelper.getTokenizer().getVocab())
                .task(BertIterator.Task.SEQ_CLASSIFICATION)
                .build();
        mds = b.next();
        assertEquals(2, mds.getFeatures().length);
        //Segment ID should be all 0s for single segment task
        INDArray segmentId = expM.like();
        assertEquals(segmentId, mds.getFeatures(1));
        assertEquals(segmentId, b.featurizeSentences(testHelper.getSentences()).getFirst()[1]);
    }

    @Test(timeout = 20000L)
    public void testBertUnsupervised() throws Exception {
        int minibatchSize = 2;
        TestSentenceHelper testHelper = new TestSentenceHelper();
        //Task 1: Unsupervised
        BertIterator b = BertIterator.builder()
                .tokenizer(testHelper.getTokenizer())
                .lengthHandling(BertIterator.LengthHandling.FIXED_LENGTH, 16)
                .minibatchSize(minibatchSize)
                .sentenceProvider(testHelper.getSentenceProvider())
                .featureArrays(BertIterator.FeatureArrays.INDICES_MASK)
                .vocabMap(testHelper.getTokenizer().getVocab())
                .task(BertIterator.Task.UNSUPERVISED)
                .masker(new BertMaskedLMMasker(new Random(12345), 0.2, 0.5, 0.5))
                .unsupervisedLabelFormat(BertIterator.UnsupervisedLabelFormat.RANK2_IDX)
                .maskToken("[MASK]")
                .build();

        System.out.println("Mask token index: " + testHelper.getTokenizer().getVocab().get("[MASK]"));

        MultiDataSet mds = b.next();
        System.out.println(mds.getFeatures(0));
        System.out.println(mds.getFeaturesMaskArray(0));
        System.out.println(mds.getLabels(0));
        System.out.println(mds.getLabelsMaskArray(0));

        assertFalse(b.hasNext());
        b.reset();
        assertTrue(b.hasNext());
    }

    @Test(timeout = 20000L)
    public void testLengthHandling() throws Exception {
        int minibatchSize = 2;
        TestSentenceHelper testHelper = new TestSentenceHelper();
        INDArray expF = Nd4j.create(DataType.INT, 1, 16);
        INDArray expM = Nd4j.create(DataType.INT, 1, 16);
        Map<String, Integer> m = testHelper.getTokenizer().getVocab();
        for (int i = 0; i < minibatchSize; i++) {
            List<String> tokens = testHelper.getTokenizedSentences().get(i);
            INDArray expFTemp = Nd4j.create(DataType.INT, 1, 16);
            INDArray expMTemp = Nd4j.create(DataType.INT, 1, 16);
            System.out.println(tokens);
            for (int j = 0; j < tokens.size(); j++) {
                String token = tokens.get(j);
                if (!m.containsKey(token)) {
                    throw new IllegalStateException("Unknown token: \"" + token + "\"");
                }
                int idx = m.get(token);
                expFTemp.putScalar(0, j, idx);
                expMTemp.putScalar(0, j, 1);
            }
            if (i == 0) {
                expF = expFTemp.dup();
                expM = expMTemp.dup();
            } else {
                expF = Nd4j.vstack(expF, expFTemp);
                expM = Nd4j.vstack(expM, expMTemp);
            }
        }

        //--------------------------------------------------------------

        //Fixed length: clip or pad - already tested in other tests

        //Any length: as long as we need to fit longest sequence

        BertIterator b = BertIterator.builder()
                .tokenizer(testHelper.getTokenizer())
                .lengthHandling(BertIterator.LengthHandling.ANY_LENGTH, -1)
                .minibatchSize(minibatchSize)
                .sentenceProvider(testHelper.getSentenceProvider())
                .featureArrays(BertIterator.FeatureArrays.INDICES_MASK)
                .vocabMap(testHelper.getTokenizer().getVocab())
                .task(BertIterator.Task.SEQ_CLASSIFICATION)
                .build();
        MultiDataSet mds = b.next();
        long[] expShape = new long[]{2, 14};
        assertArrayEquals(expShape, mds.getFeatures(0).shape());
        assertArrayEquals(expShape, mds.getFeaturesMaskArray(0).shape());
        assertEquals(expF.get(NDArrayIndex.all(), NDArrayIndex.interval(0, 14)), mds.getFeatures(0));
        assertEquals(expM.get(NDArrayIndex.all(), NDArrayIndex.interval(0, 14)), mds.getFeaturesMaskArray(0));
        assertEquals(mds.getFeatures(0), b.featurizeSentences(testHelper.getSentences()).getFirst()[0]);
        assertEquals(mds.getFeaturesMaskArray(0), b.featurizeSentences(testHelper.getSentences()).getSecond()[0]);

        //Clip only: clip to maximum, but don't pad if less
        b = BertIterator.builder()
                .tokenizer(testHelper.getTokenizer())
                .lengthHandling(BertIterator.LengthHandling.CLIP_ONLY, 20)
                .minibatchSize(minibatchSize)
                .sentenceProvider(testHelper.getSentenceProvider())
                .featureArrays(BertIterator.FeatureArrays.INDICES_MASK)
                .vocabMap(testHelper.getTokenizer().getVocab())
                .task(BertIterator.Task.SEQ_CLASSIFICATION)
                .build();
        expShape = new long[]{2, 14};
        assertArrayEquals(expShape, mds.getFeatures(0).shape());
        assertArrayEquals(expShape, mds.getFeaturesMaskArray(0).shape());
    }

    @Test(timeout = 20000L)
    public void testMinibatchPadding() throws Exception {
        Nd4j.setDefaultDataTypes(DataType.FLOAT, DataType.FLOAT);
        int minibatchSize = 3;
        TestSentenceHelper testHelper = new TestSentenceHelper(minibatchSize);
        INDArray zeros = Nd4j.create(DataType.INT, 1, 16);
        INDArray expF = Nd4j.create(DataType.INT, 1, 16);
        INDArray expM = Nd4j.create(DataType.INT, 1, 16);
        Map<String, Integer> m = testHelper.getTokenizer().getVocab();
        for (int i = 0; i < minibatchSize; i++) {
            List<String> tokens = testHelper.getTokenizedSentences().get(i);
            INDArray expFTemp = Nd4j.create(DataType.INT, 1, 16);
            INDArray expMTemp = Nd4j.create(DataType.INT, 1, 16);
            System.out.println(tokens);
            for (int j = 0; j < tokens.size(); j++) {
                String token = tokens.get(j);
                if (!m.containsKey(token)) {
                    throw new IllegalStateException("Unknown token: \"" + token + "\"");
                }
                int idx = m.get(token);
                expFTemp.putScalar(0, j, idx);
                expMTemp.putScalar(0, j, 1);
            }
            if (i == 0) {
                expF = expFTemp.dup();
                expM = expMTemp.dup();
            } else {
                expF = Nd4j.vstack(expF.dup(), expFTemp);
                expM = Nd4j.vstack(expM.dup(), expMTemp);
            }
        }

        expF = Nd4j.vstack(expF, zeros);
        expM = Nd4j.vstack(expM, zeros);
        INDArray expL = Nd4j.createFromArray(new float[][]{{0, 1}, {1, 0}, {0, 1}, {0, 0}});
        INDArray expLM = Nd4j.create(DataType.FLOAT, 4, 1);
        expLM.putScalar(0, 0, 1);
        expLM.putScalar(1, 0, 1);
        expLM.putScalar(2, 0, 1);

        //--------------------------------------------------------------

        BertIterator b = BertIterator.builder()
                .tokenizer(testHelper.getTokenizer())
                .lengthHandling(BertIterator.LengthHandling.FIXED_LENGTH, 16)
                .minibatchSize(minibatchSize + 1)
                .padMinibatches(true)
                .sentenceProvider(testHelper.getSentenceProvider())
                .featureArrays(BertIterator.FeatureArrays.INDICES_MASK_SEGMENTID)
                .vocabMap(testHelper.getTokenizer().getVocab())
                .task(BertIterator.Task.SEQ_CLASSIFICATION)
                .build();

        MultiDataSet mds = b.next();
        long[] expShape = {4, 16};
        assertArrayEquals(expShape, mds.getFeatures(0).shape());
        assertArrayEquals(expShape, mds.getFeatures(1).shape());
        assertArrayEquals(expShape, mds.getFeaturesMaskArray(0).shape());

        long[] lShape = {4, 2};
        long[] lmShape = {4, 1};
        assertArrayEquals(lShape, mds.getLabels(0).shape());
        assertArrayEquals(lmShape, mds.getLabelsMaskArray(0).shape());

        assertEquals(expF, mds.getFeatures(0));
        assertEquals(expM, mds.getFeaturesMaskArray(0));
        assertEquals(expL, mds.getLabels(0));
        assertEquals(expLM, mds.getLabelsMaskArray(0));

        assertEquals(expF, b.featurizeSentences(testHelper.getSentences()).getFirst()[0]);
        assertEquals(expM, b.featurizeSentences(testHelper.getSentences()).getSecond()[0]);
    }

    /*
        Checks that a mds from a pair sentence is equal to hstack'd mds from the left side and right side of the pair
        Checks different lengths for max length to check popping and padding
     */
    @Test
    public void testSentencePairsSingle() throws IOException {
        boolean prependAppend;
        int numOfSentences;

        TestSentenceHelper testHelper = new TestSentenceHelper();
        int shortL = testHelper.getShortestL();
        int longL = testHelper.getLongestL();

        Triple<MultiDataSet, MultiDataSet, MultiDataSet> multiDataSetTriple;
        MultiDataSet fromPair, leftSide, rightSide;

        // check for pair max length exactly equal to sum of lengths - pop neither no padding
        // should be the same as hstack with segment ids 1 for second sentence
        prependAppend = true;
        numOfSentences = 1;
        multiDataSetTriple = generateMultiDataSets(new Triple<>(shortL + longL, shortL, longL), prependAppend, numOfSentences);
        fromPair = multiDataSetTriple.getFirst();
        leftSide = multiDataSetTriple.getSecond();
        rightSide = multiDataSetTriple.getThird();
        assertEquals(fromPair.getFeatures(0), Nd4j.hstack(leftSide.getFeatures(0), rightSide.getFeatures(0)));
        rightSide.getFeatures(1).addi(1); //add 1 for right side segment ids
        assertEquals(fromPair.getFeatures(1), Nd4j.hstack(leftSide.getFeatures(1), rightSide.getFeatures(1)));
        assertEquals(fromPair.getFeaturesMaskArray(0), Nd4j.hstack(leftSide.getFeaturesMaskArray(0), rightSide.getFeaturesMaskArray(0)));

        //check for pair max length greater than sum of lengths - pop neither with padding
        // features should be the same as hstack of shorter and longer padded with prepend/append
        // segment id should 1 only in the longer for part of the length of the sentence
        prependAppend = true;
        numOfSentences = 1;
        multiDataSetTriple = generateMultiDataSets(new Triple<>(shortL + longL + 5, shortL, longL + 5), prependAppend, numOfSentences);
        fromPair = multiDataSetTriple.getFirst();
        leftSide = multiDataSetTriple.getSecond();
        rightSide = multiDataSetTriple.getThird();
        assertEquals(fromPair.getFeatures(0), Nd4j.hstack(leftSide.getFeatures(0), rightSide.getFeatures(0)));
        rightSide.getFeatures(1).get(NDArrayIndex.all(), NDArrayIndex.interval(0, longL + 1)).addi(1); //segmentId stays 0 for the padded part
        assertEquals(fromPair.getFeatures(1), Nd4j.hstack(leftSide.getFeatures(1), rightSide.getFeatures(1)));
        assertEquals(fromPair.getFeaturesMaskArray(0), Nd4j.hstack(leftSide.getFeaturesMaskArray(0), rightSide.getFeaturesMaskArray(0)));

        //check for pair max length less than shorter sentence - pop both
        //should be the same as hstack with segment ids 1 for second sentence if no prepend/append
        int maxL = 5;//checking odd
        numOfSentences = 3;
        prependAppend = false;
        multiDataSetTriple = generateMultiDataSets(new Triple<>(maxL, maxL / 2, maxL - maxL / 2), prependAppend, numOfSentences);
        fromPair = multiDataSetTriple.getFirst();
        leftSide = multiDataSetTriple.getSecond();
        rightSide = multiDataSetTriple.getThird();
        assertEquals(fromPair.getFeatures(0), Nd4j.hstack(leftSide.getFeatures(0), rightSide.getFeatures(0)));
        rightSide.getFeatures(1).addi(1);
        assertEquals(fromPair.getFeatures(1), Nd4j.hstack(leftSide.getFeatures(1), rightSide.getFeatures(1)));
        assertEquals(fromPair.getFeaturesMaskArray(0), Nd4j.hstack(leftSide.getFeaturesMaskArray(0), rightSide.getFeaturesMaskArray(0)));
    }

    /*
        Same idea as previous test - construct mds from bert iterator with sep sentences and check against one with pairs
        Checks various max lengths
        Has sentences of varying lengths
    */
    @Test
    public void testSentencePairsUnequalLengths() throws IOException {

        int minibatchSize = 4;
        int numOfSentencesinIter = 3;

        TestSentencePairsHelper testPairHelper = new TestSentencePairsHelper(numOfSentencesinIter);
        int shortL = testPairHelper.getShortL();
        int longL = testPairHelper.getLongL();
        int sent1L = testPairHelper.getSentenceALen();
        int sent2L = testPairHelper.getSentenceBLen();

        System.out.println("Sentence Pairs, Left");
        System.out.println(testPairHelper.getSentencesLeft());
        System.out.println("Sentence Pairs, Right");
        System.out.println(testPairHelper.getSentencesRight());

        //anything outside this range more will need to check padding,truncation
        for (int maxL = longL + shortL; maxL > 2 * shortL + 1; maxL--) {

            System.out.println("Running for max length = " + maxL);

            MultiDataSet leftMDS = BertIterator.builder()
                    .tokenizer(testPairHelper.getTokenizer())
                    .minibatchSize(minibatchSize)
                    .featureArrays(BertIterator.FeatureArrays.INDICES_MASK_SEGMENTID)
                    .vocabMap(testPairHelper.getTokenizer().getVocab())
                    .task(BertIterator.Task.SEQ_CLASSIFICATION)
                    .lengthHandling(BertIterator.LengthHandling.FIXED_LENGTH, longL * 10) //random big num guaranteed to be longer than either
                    .sentenceProvider(new TestSentenceHelper(numOfSentencesinIter).getSentenceProvider())
                    .padMinibatches(true)
                    .build().next();

            MultiDataSet rightMDS = BertIterator.builder()
                    .tokenizer(testPairHelper.getTokenizer())
                    .minibatchSize(minibatchSize)
                    .featureArrays(BertIterator.FeatureArrays.INDICES_MASK_SEGMENTID)
                    .vocabMap(testPairHelper.getTokenizer().getVocab())
                    .task(BertIterator.Task.SEQ_CLASSIFICATION)
                    .lengthHandling(BertIterator.LengthHandling.FIXED_LENGTH, longL * 10) //random big num guaranteed to be longer than either
                    .sentenceProvider(new TestSentenceHelper(true, numOfSentencesinIter).getSentenceProvider())
                    .padMinibatches(true)
                    .build().next();

            MultiDataSet pairMDS = BertIterator.builder()
                    .tokenizer(testPairHelper.getTokenizer())
                    .minibatchSize(minibatchSize)
                    .featureArrays(BertIterator.FeatureArrays.INDICES_MASK_SEGMENTID)
                    .vocabMap(testPairHelper.getTokenizer().getVocab())
                    .task(BertIterator.Task.SEQ_CLASSIFICATION)
                    .lengthHandling(BertIterator.LengthHandling.FIXED_LENGTH, maxL)
                    .sentencePairProvider(testPairHelper.getPairSentenceProvider())
                    .padMinibatches(true)
                    .build().next();

            //CHECK FEATURES
            INDArray combinedFeat = Nd4j.create(DataType.INT, minibatchSize, maxL);
            //left side
            INDArray leftFeatures = leftMDS.getFeatures(0);
            INDArray topLSentFeat = leftFeatures.getRow(0).get(NDArrayIndex.interval(0, shortL));
            INDArray midLSentFeat = leftFeatures.getRow(1).get(NDArrayIndex.interval(0, maxL - shortL));
            INDArray bottomLSentFeat = leftFeatures.getRow(2).get(NDArrayIndex.interval(0, sent1L));
            //right side
            INDArray rightFeatures = rightMDS.getFeatures(0);
            INDArray topRSentFeat = rightFeatures.getRow(0).get(NDArrayIndex.interval(0, maxL - shortL));
            INDArray midRSentFeat = rightFeatures.getRow(1).get(NDArrayIndex.interval(0, shortL));
            INDArray bottomRSentFeat = rightFeatures.getRow(2).get(NDArrayIndex.interval(0, sent2L));
            //expected pair
            combinedFeat.getRow(0).addi(Nd4j.hstack(topLSentFeat, topRSentFeat));
            combinedFeat.getRow(1).addi(Nd4j.hstack(midLSentFeat, midRSentFeat));
            combinedFeat.getRow(2).get(NDArrayIndex.interval(0, sent1L + sent2L)).addi(Nd4j.hstack(bottomLSentFeat, bottomRSentFeat));

            assertEquals(maxL, pairMDS.getFeatures(0).shape()[1]);
            assertArrayEquals(combinedFeat.shape(), pairMDS.getFeatures(0).shape());
            assertEquals(combinedFeat, pairMDS.getFeatures(0));

            //CHECK SEGMENT ID
            INDArray combinedFetSeg = Nd4j.create(DataType.INT, minibatchSize, maxL);
            combinedFetSeg.get(NDArrayIndex.point(0), NDArrayIndex.interval(shortL, maxL)).addi(1);
            combinedFetSeg.get(NDArrayIndex.point(1), NDArrayIndex.interval(maxL - shortL, maxL)).addi(1);
            combinedFetSeg.get(NDArrayIndex.point(2), NDArrayIndex.interval(sent1L, sent1L + sent2L)).addi(1);
            assertArrayEquals(combinedFetSeg.shape(), pairMDS.getFeatures(1).shape());
            assertEquals(maxL, combinedFetSeg.shape()[1]);
            assertEquals(combinedFetSeg, pairMDS.getFeatures(1));

            testPairHelper.getPairSentenceProvider().reset();
        }
    }

    @Test
    public void testSentencePairFeaturizer() throws IOException {
        int minibatchSize = 2;
        TestSentencePairsHelper testPairHelper = new TestSentencePairsHelper(minibatchSize);
        BertIterator b = BertIterator.builder()
                .tokenizer(testPairHelper.getTokenizer())
                .minibatchSize(minibatchSize)
                .padMinibatches(true)
                .featureArrays(BertIterator.FeatureArrays.INDICES_MASK_SEGMENTID)
                .vocabMap(testPairHelper.getTokenizer().getVocab())
                .task(BertIterator.Task.SEQ_CLASSIFICATION)
                .lengthHandling(BertIterator.LengthHandling.FIXED_LENGTH, 128)
                .sentencePairProvider(testPairHelper.getPairSentenceProvider())
                .prependToken("[CLS]")
                .appendToken("[SEP]")
                .build();
        MultiDataSet mds = b.next();
        INDArray[] featuresArr = mds.getFeatures();
        INDArray[] featuresMaskArr = mds.getFeaturesMaskArrays();

        Pair<INDArray[], INDArray[]> p = b.featurizeSentencePairs(testPairHelper.getSentencePairs());
        assertEquals(p.getFirst().length, 2);
        assertEquals(featuresArr[0], p.getFirst()[0]);
        assertEquals(featuresArr[1], p.getFirst()[1]);
        assertEquals(featuresMaskArr[0], p.getSecond()[0]);
    }

    /**
     * Returns three multidatasets (one from pair of sentences and the other two from single sentence lists) from bert iterator
     * with given max lengths and whether to prepend/append
     * Idea is the sentence pair dataset can be constructed from the single sentence datasets
     */
    private Triple<MultiDataSet, MultiDataSet, MultiDataSet> generateMultiDataSets(Triple<Integer, Integer, Integer> maxLengths, boolean prependAppend, int numSentences) throws IOException {
        BertWordPieceTokenizerFactory t = new BertWordPieceTokenizerFactory(pathToVocab, false, false, c);
        int maxforPair = maxLengths.getFirst();
        int maxPartOne = maxLengths.getSecond();
        int maxPartTwo = maxLengths.getThird();
        BertIterator.Builder commonBuilder;
        commonBuilder = BertIterator.builder()
                .tokenizer(t)
                .minibatchSize(4)
                .featureArrays(BertIterator.FeatureArrays.INDICES_MASK_SEGMENTID)
                .vocabMap(t.getVocab())
                .task(BertIterator.Task.SEQ_CLASSIFICATION);
        BertIterator pairIter = commonBuilder
                .lengthHandling(BertIterator.LengthHandling.FIXED_LENGTH, prependAppend ? maxforPair + 3 : maxforPair)
                .sentencePairProvider(new TestSentencePairsHelper(numSentences).getPairSentenceProvider())
                .prependToken(prependAppend ? "[CLS]" : null)
                .appendToken(prependAppend ? "[SEP]" : null)
                .build();
        BertIterator leftIter = commonBuilder
                .lengthHandling(BertIterator.LengthHandling.FIXED_LENGTH, prependAppend ? maxPartOne + 2 : maxPartOne)
                .sentenceProvider(new TestSentenceHelper(numSentences).getSentenceProvider())
                .prependToken(prependAppend ? "[CLS]" : null)
                .appendToken(prependAppend ? "[SEP]" : null)
                .build();
        BertIterator rightIter = commonBuilder
                .lengthHandling(BertIterator.LengthHandling.FIXED_LENGTH, prependAppend ? maxPartTwo + 1 : maxPartTwo)
                .sentenceProvider(new TestSentenceHelper(true, numSentences).getSentenceProvider())
                .prependToken(null)
                .appendToken(prependAppend ? "[SEP]" : null)
                .build();
        return new Triple<>(pairIter.next(), leftIter.next(), rightIter.next());
    }

    @Getter
    private static class TestSentencePairsHelper {

        private List<String> sentencesLeft;
        private List<String> sentencesRight;
        private List<Pair<String, String>> sentencePairs;
        private List<List<String>> tokenizedSentencesLeft;
        private List<List<String>> tokenizedSentencesRight;
        private List<String> labels;
        private int shortL;
        private int longL;
        private int sentenceALen;
        private int sentenceBLen;
        private BertWordPieceTokenizerFactory tokenizer;
        private CollectionLabeledPairSentenceProvider pairSentenceProvider;

        private TestSentencePairsHelper() throws IOException {
            this(3);
        }

        private TestSentencePairsHelper(int minibatchSize) throws IOException {
            sentencesLeft = new ArrayList<>();
            sentencesRight = new ArrayList<>();
            sentencePairs = new ArrayList<>();
            labels = new ArrayList<>();
            tokenizedSentencesLeft = new ArrayList<>();
            tokenizedSentencesRight = new ArrayList<>();
            tokenizer = new BertWordPieceTokenizerFactory(pathToVocab, false, false, c);
            sentencesLeft.add(shortSentence);
            sentencesRight.add(longSentence);
            sentencePairs.add(new Pair<>(shortSentence, longSentence));
            labels.add("positive");
            if (minibatchSize > 1) {
                sentencesLeft.add(longSentence);
                sentencesRight.add(shortSentence);
                sentencePairs.add(new Pair<>(longSentence, shortSentence));
                labels.add("negative");
                if (minibatchSize > 2) {
                    sentencesLeft.add(sentenceA);
                    sentencesRight.add(sentenceB);
                    sentencePairs.add(new Pair<>(sentenceA, sentenceB));
                    labels.add("positive");
                }
            }
            for (int i = 0; i < minibatchSize; i++) {
                List<String> tokensL = tokenizer.create(sentencesLeft.get(i)).getTokens();
                List<String> tokensR = tokenizer.create(sentencesRight.get(i)).getTokens();
                if (i == 0) {
                    shortL = tokensL.size();
                    longL = tokensR.size();
                }
                if (i == 2) {
                    sentenceALen = tokensL.size();
                    sentenceBLen = tokensR.size();
                }
                tokenizedSentencesLeft.add(tokensL);
                tokenizedSentencesRight.add(tokensR);
            }
            pairSentenceProvider = new CollectionLabeledPairSentenceProvider(sentencesLeft, sentencesRight, labels, null);
        }
    }

    @Getter
    private static class TestSentenceHelper {

        private List<String> sentences;
        private List<List<String>> tokenizedSentences;
        private List<String> labels;
        private int shortestL = 0;
        private int longestL = 0;
        private BertWordPieceTokenizerFactory tokenizer;
        private CollectionLabeledSentenceProvider sentenceProvider;

        private TestSentenceHelper() throws IOException {
            this(false, 2);
        }

        private TestSentenceHelper(int minibatchSize) throws IOException {
            this(false, minibatchSize);
        }

        private TestSentenceHelper(boolean alternateOrder) throws IOException {
            this(false, 3);
        }

        private TestSentenceHelper(boolean alternateOrder, int minibatchSize) throws IOException {
            sentences = new ArrayList<>();
            labels = new ArrayList<>();
            tokenizedSentences = new ArrayList<>();
            tokenizer = new BertWordPieceTokenizerFactory(pathToVocab, false, false, c);
            if (!alternateOrder) {
                sentences.add(shortSentence);
                labels.add("positive");
                if (minibatchSize > 1) {
                    sentences.add(longSentence);
                    labels.add("negative");
                    if (minibatchSize > 2) {
                        sentences.add(sentenceA);
                        labels.add("positive");
                    }
                }
            } else {
                sentences.add(longSentence);
                labels.add("negative");
                if (minibatchSize > 1) {
                    sentences.add(shortSentence);
                    labels.add("positive");
                    if (minibatchSize > 2) {
                        sentences.add(sentenceB);
                        labels.add("positive");
                    }
                }
            }
            for (int i = 0; i < sentences.size(); i++) {
                List<String> tokenizedSentence = tokenizer.create(sentences.get(i)).getTokens();
                if (i == 0)
                    shortestL = tokenizedSentence.size();
                if (tokenizedSentence.size() > longestL)
                    longestL = tokenizedSentence.size();
                if (tokenizedSentence.size() < shortestL)
                    shortestL = tokenizedSentence.size();
                tokenizedSentences.add(tokenizedSentence);
            }
            sentenceProvider = new CollectionLabeledSentenceProvider(sentences, labels, null);
        }
    }

}
