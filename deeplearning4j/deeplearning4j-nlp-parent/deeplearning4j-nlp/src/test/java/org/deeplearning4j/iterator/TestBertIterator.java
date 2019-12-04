/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
 * Copyright (c) 2019 Konduit K.K.
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

package org.deeplearning4j.iterator;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.iterator.bert.BertMaskedLMMasker;
import org.deeplearning4j.text.tokenization.tokenizerfactory.BertWordPieceTokenizerFactory;
import org.junit.Test;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.primitives.Triple;
import org.nd4j.resources.Resources;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.util.*;

import static org.junit.Assert.*;


public class TestBertIterator extends BaseDL4JTest {

    private File pathToVocab = Resources.asFile("other/vocab.txt");
    private static Charset c = StandardCharsets.UTF_8;

    public TestBertIterator() throws IOException {
    }

    @Test(timeout = 20000L)
    public void testBertSequenceClassification() throws Exception {

        String toTokenize1 = "I saw a girl with a telescope.";
        String toTokenize2 = "Donaudampfschifffahrts Kapitänsmützeninnenfuttersaum";
        List<String> forInference = new ArrayList<>();
        forInference.add(toTokenize1);
        forInference.add(toTokenize2);
        BertWordPieceTokenizerFactory t = new BertWordPieceTokenizerFactory(pathToVocab, false, false, c);

        BertIterator b = BertIterator.builder()
                .tokenizer(t)
                .lengthHandling(BertIterator.LengthHandling.FIXED_LENGTH, 16)
                .minibatchSize(2)
                .sentenceProvider(new TestSentenceProvider())
                .featureArrays(BertIterator.FeatureArrays.INDICES_MASK)
                .vocabMap(t.getVocab())
                .task(BertIterator.Task.SEQ_CLASSIFICATION)
                .build();

        MultiDataSet mds = b.next();
        assertEquals(1, mds.getFeatures().length);
        System.out.println(mds.getFeatures(0));
        System.out.println(mds.getFeaturesMaskArray(0));


        INDArray expEx0 = Nd4j.create(DataType.INT, 1, 16);
        INDArray expM0 = Nd4j.create(DataType.INT, 1, 16);
        List<String> tokens = t.create(toTokenize1).getTokens();
        Map<String, Integer> m = t.getVocab();
        for (int i = 0; i < tokens.size(); i++) {
            int idx = m.get(tokens.get(i));
            expEx0.putScalar(0, i, idx);
            expM0.putScalar(0, i, 1);
        }

        INDArray expEx1 = Nd4j.create(DataType.INT, 1, 16);
        INDArray expM1 = Nd4j.create(DataType.INT, 1, 16);
        List<String> tokens2 = t.create(toTokenize2).getTokens();
        for (int i = 0; i < tokens2.size(); i++) {
            String token = tokens2.get(i);
            if (!m.containsKey(token)) {
                throw new IllegalStateException("Unknown token: \"" + token + "\"");
            }
            int idx = m.get(token);
            expEx1.putScalar(0, i, idx);
            expM1.putScalar(0, i, 1);
        }

        INDArray expF = Nd4j.vstack(expEx0, expEx1);
        INDArray expM = Nd4j.vstack(expM0, expM1);

        assertEquals(expF, mds.getFeatures(0));
        assertEquals(expM, mds.getFeaturesMaskArray(0));
        assertEquals(expF, b.featurizeSentences(forInference).getFirst()[0]);
        assertEquals(expM, b.featurizeSentences(forInference).getSecond()[0]);

        b.next(); //pop the third element
        assertFalse(b.hasNext());
        b.reset();
        assertTrue(b.hasNext());

        forInference.set(0, toTokenize2);
        //Same thing, but with segment ID also
        b = BertIterator.builder()
                .tokenizer(t)
                .lengthHandling(BertIterator.LengthHandling.FIXED_LENGTH, 16)
                .minibatchSize(2)
                .sentenceProvider(new TestSentenceProvider())
                .featureArrays(BertIterator.FeatureArrays.INDICES_MASK_SEGMENTID)
                .vocabMap(t.getVocab())
                .task(BertIterator.Task.SEQ_CLASSIFICATION)
                .build();
        mds = b.next();
        assertEquals(2, mds.getFeatures().length);
        //assertEquals(2, mds.getFeaturesMaskArrays().length); second element is null...
        assertEquals(2, b.featurizeSentences(forInference).getFirst().length);
        //Segment ID should be all 0s for single segment task
        INDArray segmentId = expM.like();
        assertEquals(segmentId, mds.getFeatures(1));
        assertEquals(segmentId, b.featurizeSentences(forInference).getFirst()[1]);
    }

    @Test(timeout = 20000L)
    public void testBertUnsupervised() throws Exception {
        //Task 1: Unsupervised
        BertWordPieceTokenizerFactory t = new BertWordPieceTokenizerFactory(pathToVocab, false, false, c);
        BertIterator b = BertIterator.builder()
                .tokenizer(t)
                .lengthHandling(BertIterator.LengthHandling.FIXED_LENGTH, 16)
                .minibatchSize(2)
                .sentenceProvider(new TestSentenceProvider())
                .featureArrays(BertIterator.FeatureArrays.INDICES_MASK)
                .vocabMap(t.getVocab())
                .task(BertIterator.Task.UNSUPERVISED)
                .masker(new BertMaskedLMMasker(new Random(12345), 0.2, 0.5, 0.5))
                .unsupervisedLabelFormat(BertIterator.UnsupervisedLabelFormat.RANK2_IDX)
                .maskToken("[MASK]")
                .build();

        System.out.println("Mask token index: " + t.getVocab().get("[MASK]"));

        MultiDataSet mds = b.next();
        System.out.println(mds.getFeatures(0));
        System.out.println(mds.getFeaturesMaskArray(0));
        System.out.println(mds.getLabels(0));
        System.out.println(mds.getLabelsMaskArray(0));

        b.next(); //pop the third element
        assertFalse(b.hasNext());
        b.reset();
        assertTrue(b.hasNext());
    }

    @Test(timeout = 20000L)
    public void testLengthHandling() throws Exception {
        String toTokenize1 = "I saw a girl with a telescope.";
        String toTokenize2 = "Donaudampfschifffahrts Kapitänsmützeninnenfuttersaum";
        List<String> forInference = new ArrayList<>();
        forInference.add(toTokenize1);
        forInference.add(toTokenize2);
        BertWordPieceTokenizerFactory t = new BertWordPieceTokenizerFactory(pathToVocab, false, false, c);
        INDArray expEx0 = Nd4j.create(DataType.INT, 1, 16);
        INDArray expM0 = Nd4j.create(DataType.INT, 1, 16);
        List<String> tokens = t.create(toTokenize1).getTokens();
        System.out.println(tokens);
        Map<String, Integer> m = t.getVocab();
        for (int i = 0; i < tokens.size(); i++) {
            int idx = m.get(tokens.get(i));
            expEx0.putScalar(0, i, idx);
            expM0.putScalar(0, i, 1);
        }

        INDArray expEx1 = Nd4j.create(DataType.INT, 1, 16);
        INDArray expM1 = Nd4j.create(DataType.INT, 1, 16);
        List<String> tokens2 = t.create(toTokenize2).getTokens();
        System.out.println(tokens2);
        for (int i = 0; i < tokens2.size(); i++) {
            String token = tokens2.get(i);
            if (!m.containsKey(token)) {
                throw new IllegalStateException("Unknown token: \"" + token + "\"");
            }
            int idx = m.get(token);
            expEx1.putScalar(0, i, idx);
            expM1.putScalar(0, i, 1);
        }

        INDArray expF = Nd4j.vstack(expEx0, expEx1);
        INDArray expM = Nd4j.vstack(expM0, expM1);

        //--------------------------------------------------------------

        //Fixed length: clip or pad - already tested in other tests

        //Any length: as long as we need to fit longest sequence

        BertIterator b = BertIterator.builder()
                .tokenizer(t)
                .lengthHandling(BertIterator.LengthHandling.ANY_LENGTH, -1)
                .minibatchSize(2)
                .sentenceProvider(new TestSentenceProvider())
                .featureArrays(BertIterator.FeatureArrays.INDICES_MASK)
                .vocabMap(t.getVocab())
                .task(BertIterator.Task.SEQ_CLASSIFICATION)
                .build();
        MultiDataSet mds = b.next();
        long[] expShape = new long[]{2, 14};
        assertArrayEquals(expShape, mds.getFeatures(0).shape());
        assertArrayEquals(expShape, mds.getFeaturesMaskArray(0).shape());
        assertEquals(expF.get(NDArrayIndex.all(), NDArrayIndex.interval(0, 14)), mds.getFeatures(0));
        assertEquals(expM.get(NDArrayIndex.all(), NDArrayIndex.interval(0, 14)), mds.getFeaturesMaskArray(0));
        assertEquals(mds.getFeatures(0), b.featurizeSentences(forInference).getFirst()[0]);
        assertEquals(mds.getFeaturesMaskArray(0), b.featurizeSentences(forInference).getSecond()[0]);

        //Clip only: clip to maximum, but don't pad if less
        b = BertIterator.builder()
                .tokenizer(t)
                .lengthHandling(BertIterator.LengthHandling.CLIP_ONLY, 20)
                .minibatchSize(2)
                .sentenceProvider(new TestSentenceProvider())
                .featureArrays(BertIterator.FeatureArrays.INDICES_MASK)
                .vocabMap(t.getVocab())
                .task(BertIterator.Task.SEQ_CLASSIFICATION)
                .build();
        mds = b.next();
        expShape = new long[]{2, 14};
        assertArrayEquals(expShape, mds.getFeatures(0).shape());
        assertArrayEquals(expShape, mds.getFeaturesMaskArray(0).shape());
    }

    @Test(timeout = 20000L)
    public void testMinibatchPadding() throws Exception {
        Nd4j.setDefaultDataTypes(DataType.FLOAT, DataType.FLOAT);
        String toTokenize1 = "I saw a girl with a telescope.";
        String toTokenize2 = "Donaudampfschifffahrts Kapitänsmützeninnenfuttersaum";
        String toTokenize3 = "Goodnight noises everywhere";
        List<String> forInference = new ArrayList<>();
        forInference.add(toTokenize1);
        forInference.add(toTokenize2);
        forInference.add(toTokenize3);
        BertWordPieceTokenizerFactory t = new BertWordPieceTokenizerFactory(pathToVocab, false, false, c);
        INDArray expEx0 = Nd4j.create(DataType.INT, 1, 16);
        INDArray expM0 = Nd4j.create(DataType.INT, 1, 16);
        List<String> tokens = t.create(toTokenize1).getTokens();
        Map<String, Integer> m = t.getVocab();
        for (int i = 0; i < tokens.size(); i++) {
            int idx = m.get(tokens.get(i));
            expEx0.putScalar(0, i, idx);
            expM0.putScalar(0, i, 1);
        }

        INDArray expEx1 = Nd4j.create(DataType.INT, 1, 16);
        INDArray expM1 = Nd4j.create(DataType.INT, 1, 16);
        List<String> tokens2 = t.create(toTokenize2).getTokens();
        for (int i = 0; i < tokens2.size(); i++) {
            String token = tokens2.get(i);
            if (!m.containsKey(token)) {
                throw new IllegalStateException("Unknown token: \"" + token + "\"");
            }
            int idx = m.get(token);
            expEx1.putScalar(0, i, idx);
            expM1.putScalar(0, i, 1);
        }

        INDArray expEx3 = Nd4j.create(DataType.INT, 1, 16);
        INDArray expM3 = Nd4j.create(DataType.INT, 1, 16);
        List<String> tokens3 = t.create(toTokenize3).getTokens();
        for (int i = 0; i < tokens3.size(); i++) {
            String token = tokens3.get(i);
            if (!m.containsKey(token)) {
                throw new IllegalStateException("Unknown token: \"" + token + "\"");
            }
            int idx = m.get(token);
            expEx3.putScalar(0, i, idx);
            expM3.putScalar(0, i, 1);
        }

        INDArray zeros = Nd4j.create(DataType.INT, 1, 16);
        INDArray expF = Nd4j.vstack(expEx0, expEx1, expEx3, zeros);
        INDArray expM = Nd4j.vstack(expM0, expM1, expM3, zeros);
        INDArray expL = Nd4j.createFromArray(new float[][]{{1, 0}, {0, 1}, {1, 0}, {0, 0}});
        INDArray expLM = Nd4j.create(DataType.FLOAT, 4, 1);
        expLM.putScalar(0, 0, 1);
        expLM.putScalar(1, 0, 1);
        expLM.putScalar(2, 0, 1);

        //--------------------------------------------------------------

        BertIterator b = BertIterator.builder()
                .tokenizer(t)
                .lengthHandling(BertIterator.LengthHandling.FIXED_LENGTH, 16)
                .minibatchSize(4)
                .padMinibatches(true)
                .sentenceProvider(new TestSentenceProvider())
                .featureArrays(BertIterator.FeatureArrays.INDICES_MASK_SEGMENTID)
                .vocabMap(t.getVocab())
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

        assertEquals(expF, b.featurizeSentences(forInference).getFirst()[0]);
        assertEquals(expM, b.featurizeSentences(forInference).getSecond()[0]);
    }

    @Test
    public void testSentencePairsSingle() throws IOException {
        String shortSent = "I saw a girl with a telescope.";
        String longSent = "Donaudampfschifffahrts Kapitänsmützeninnenfuttersaum";
        boolean prependAppend;
        BertWordPieceTokenizerFactory t = new BertWordPieceTokenizerFactory(pathToVocab, false, false, c);
        int shortL = t.create(shortSent).countTokens();
        int longL = t.create(longSent).countTokens();

        Triple<MultiDataSet, MultiDataSet, MultiDataSet> multiDataSetTriple;
        MultiDataSet shortLongPair, shortSentence, longSentence;

        // check for pair max length exactly equal to sum of lengths - pop neither no padding
        // should be the same as hstack with segment ids 1 for second sentence
        prependAppend = true;
        multiDataSetTriple = generateMultiDataSets(new Triple<>(shortL + longL, shortL, longL), prependAppend);
        shortLongPair = multiDataSetTriple.getFirst();
        shortSentence = multiDataSetTriple.getSecond();
        longSentence = multiDataSetTriple.getThird();
        assertEquals(shortLongPair.getFeatures(0), Nd4j.hstack(shortSentence.getFeatures(0), longSentence.getFeatures(0)));
        longSentence.getFeatures(1).addi(1);
        assertEquals(shortLongPair.getFeatures(1), Nd4j.hstack(shortSentence.getFeatures(1), longSentence.getFeatures(1)));
        assertEquals(shortLongPair.getFeaturesMaskArray(0), Nd4j.hstack(shortSentence.getFeaturesMaskArray(0), longSentence.getFeaturesMaskArray(0)));

        //check for pair max length greater than sum of lengths - pop neither with padding
        // features should be the same as hstack of shorter and longer padded with prepend/append
        // segment id should 1 only in the longer for part of the length of the sentence
        prependAppend = true;
        multiDataSetTriple = generateMultiDataSets(new Triple<>(shortL + longL + 5, shortL, longL + 5), prependAppend);
        shortLongPair = multiDataSetTriple.getFirst();
        shortSentence = multiDataSetTriple.getSecond();
        longSentence = multiDataSetTriple.getThird();
        assertEquals(shortLongPair.getFeatures(0), Nd4j.hstack(shortSentence.getFeatures(0), longSentence.getFeatures(0)));
        longSentence.getFeatures(1).get(NDArrayIndex.all(), NDArrayIndex.interval(0, longL + 1)).addi(1); //segmentId stays 0 for the padded part
        assertEquals(shortLongPair.getFeatures(1), Nd4j.hstack(shortSentence.getFeatures(1), longSentence.getFeatures(1)));
        assertEquals(shortLongPair.getFeaturesMaskArray(0), Nd4j.hstack(shortSentence.getFeaturesMaskArray(0), longSentence.getFeaturesMaskArray(0)));

        //check for pair max length less than shorter sentence - pop both
        //should be the same as hstack with segment ids 1 for second sentence if no prepend/append
        int maxL = shortL - 2;
        prependAppend = false;
        multiDataSetTriple = generateMultiDataSets(new Triple<>(maxL, maxL / 2, maxL - maxL / 2), prependAppend);
        shortLongPair = multiDataSetTriple.getFirst();
        shortSentence = multiDataSetTriple.getSecond();
        longSentence = multiDataSetTriple.getThird();
        assertEquals(shortLongPair.getFeatures(0), Nd4j.hstack(shortSentence.getFeatures(0), longSentence.getFeatures(0)));
        longSentence.getFeatures(1).addi(1);
        assertEquals(shortLongPair.getFeatures(1), Nd4j.hstack(shortSentence.getFeatures(1), longSentence.getFeatures(1)));
        assertEquals(shortLongPair.getFeaturesMaskArray(0), Nd4j.hstack(shortSentence.getFeaturesMaskArray(0), longSentence.getFeaturesMaskArray(0)));
    }

    @Test
    public void testSentencePairsUnequalLengths() throws IOException {
        //check for pop only longer (i.e between longer and longer + shorter), first row pop from second sentence, next row pop from first sentence, nothing to pop in the third row
        //should be identical to hstack if there is no append, prepend
        //batch size is 2
        int mbS = 4;
        String shortSent = "I saw a girl with a telescope.";
        String longSent = "Donaudampfschifffahrts Kapitänsmützeninnenfuttersaum";
        String sent1 = "Goodnight noises everywhere"; //shorter than shortSent - no popping
        String sent2 = "Goodnight moon"; //shorter than shortSent - no popping
        BertWordPieceTokenizerFactory t = new BertWordPieceTokenizerFactory(pathToVocab, false, false, c);
        int shortL = t.create(shortSent).countTokens();
        int longL = t.create(longSent).countTokens();
        int sent1L = t.create(sent1).countTokens();
        int sent2L = t.create(sent2).countTokens();
        //won't check 2*shortL + 1 because this will always pop on the left
        for (int maxL = longL + shortL - 1; maxL > 2 * shortL; maxL--) {
            MultiDataSet leftMDS = BertIterator.builder()
                    .tokenizer(t)
                    .minibatchSize(mbS)
                    .featureArrays(BertIterator.FeatureArrays.INDICES_MASK_SEGMENTID)
                    .vocabMap(t.getVocab())
                    .task(BertIterator.Task.SEQ_CLASSIFICATION)
                    .lengthHandling(BertIterator.LengthHandling.FIXED_LENGTH, longL + 10) //random big num guaranteed to be longer than either
                    .sentenceProvider(new TestSentenceProvider())
                    .padMinibatches(true)
                    .build().next();

            MultiDataSet rightMDS = BertIterator.builder()
                    .tokenizer(t)
                    .minibatchSize(mbS)
                    .featureArrays(BertIterator.FeatureArrays.INDICES_MASK_SEGMENTID)
                    .vocabMap(t.getVocab())
                    .task(BertIterator.Task.SEQ_CLASSIFICATION)
                    .lengthHandling(BertIterator.LengthHandling.FIXED_LENGTH, longL + 10) //random big num guaranteed to be longer than either
                    .sentenceProvider(new TestSentenceProvider(true))
                    .padMinibatches(true)
                    .build().next();

            MultiDataSet pairMDS = BertIterator.builder()
                    .tokenizer(t)
                    .minibatchSize(mbS)
                    .featureArrays(BertIterator.FeatureArrays.INDICES_MASK_SEGMENTID)
                    .vocabMap(t.getVocab())
                    .task(BertIterator.Task.SEQ_CLASSIFICATION)
                    .lengthHandling(BertIterator.LengthHandling.FIXED_LENGTH, maxL) //random big num guaranteed to be longer than either
                    .sentencePairProvider(new TestSentencePairProvider())
                    .padMinibatches(true)
                    .build().next();

            //Left sentences here are {{shortSent},
            //                         {longSent},
            //                         {Sent1}}
            //Right sentences here are {{longSent},
            //                         {shortSent},
            //                         {Sent2}}
            //The sentence pairs here are {{shortSent,longSent},
            //                             {longSent,shortSent}
            //                             {Sent1, Sent2}}

            //CHECK FEATURES
            INDArray combinedFeat = Nd4j.create(DataType.INT,mbS,maxL);
            //left side
            INDArray leftFeatures = leftMDS.getFeatures(0);
            INDArray topLSentFeat = leftFeatures.getRow(0).get(NDArrayIndex.interval(0, shortL));
            INDArray midLSentFeat = leftFeatures.getRow(1).get(NDArrayIndex.interval(0, maxL - shortL));
            INDArray bottomLSentFeat = leftFeatures.getRow(2).get(NDArrayIndex.interval(0,sent1L));
            //right side
            INDArray rightFeatures = rightMDS.getFeatures(0);
            INDArray topRSentFeat = rightFeatures.getRow(0).get(NDArrayIndex.interval(0, maxL - shortL));
            INDArray midRSentFeat = rightFeatures.getRow(1).get(NDArrayIndex.interval(0, shortL));
            INDArray bottomRSentFeat = rightFeatures.getRow(2).get(NDArrayIndex.interval(0,sent2L));
            //expected pair
            combinedFeat.getRow(0).addi(Nd4j.hstack(topLSentFeat,topRSentFeat));
            combinedFeat.getRow(1).addi(Nd4j.hstack(midLSentFeat,midRSentFeat));
            combinedFeat.getRow(2).get(NDArrayIndex.interval(0,sent1L+sent2L)).addi(Nd4j.hstack(bottomLSentFeat,bottomRSentFeat));

            assertEquals(maxL, pairMDS.getFeatures(0).shape()[1]);
            assertArrayEquals(combinedFeat.shape(), pairMDS.getFeatures(0).shape());
            assertEquals(combinedFeat, pairMDS.getFeatures(0));

            //CHECK SEGMENT ID
            INDArray combinedFetSeg = Nd4j.create(DataType.INT, mbS, maxL);
            combinedFetSeg.get(NDArrayIndex.point(0), NDArrayIndex.interval(shortL, maxL)).addi(1);
            combinedFetSeg.get(NDArrayIndex.point(1), NDArrayIndex.interval(maxL - shortL, maxL)).addi(1);
            combinedFetSeg.get(NDArrayIndex.point(2), NDArrayIndex.interval(sent1L, sent1L+sent2L)).addi(1);
            assertArrayEquals(combinedFetSeg.shape(), pairMDS.getFeatures(1).shape());
            assertEquals(maxL, combinedFetSeg.shape()[1]);
            assertEquals(combinedFetSeg, pairMDS.getFeatures(1));
        }
    }

    @Test
    public void testSentencePairFeaturizer() throws IOException {
        String shortSent = "I saw a girl with a telescope.";
        String longSent = "Donaudampfschifffahrts Kapitänsmützeninnenfuttersaum";
        List<Pair<String, String>> listSentencePair = new ArrayList<>();
        listSentencePair.add(new Pair<>(shortSent, longSent));
        listSentencePair.add(new Pair<>(longSent, shortSent));
        BertWordPieceTokenizerFactory t = new BertWordPieceTokenizerFactory(pathToVocab, false, false, c);
        BertIterator b = BertIterator.builder()
                .tokenizer(t)
                .minibatchSize(2)
                .padMinibatches(true)
                .featureArrays(BertIterator.FeatureArrays.INDICES_MASK_SEGMENTID)
                .vocabMap(t.getVocab())
                .task(BertIterator.Task.SEQ_CLASSIFICATION)
                .lengthHandling(BertIterator.LengthHandling.FIXED_LENGTH, 128)
                .sentencePairProvider(new TestSentencePairProvider())
                .prependToken("[CLS]")
                .appendToken("[SEP]")
                .build();
        MultiDataSet mds = b.next();
        INDArray[] featuresArr = mds.getFeatures();
        INDArray[] featuresMaskArr = mds.getFeaturesMaskArrays();

        Pair<INDArray[], INDArray[]> p = b.featurizeSentencePairs(listSentencePair);
        assertEquals(p.getFirst().length, 2);
        assertEquals(featuresArr[0], p.getFirst()[0]);
        assertEquals(featuresArr[1], p.getFirst()[1]);
        //assertEquals(p.getSecond().length, 2);
        assertEquals(featuresMaskArr[0], p.getSecond()[0]);
        //assertEquals(featuresMaskArr[1], p.getSecond()[1]);
    }

    /**
     * Returns three multidatasets from bert iterator based on given max lengths and whether to prepend/append
     * Idea is the sentence pair dataset can be constructed from the single sentence datasets
     * First one is constructed from a sentence pair "I saw a girl with a telescope." & "Donaudampfschifffahrts Kapitänsmützeninnenfuttersaum"
     * Second one is constructed from the left of the sentence pair i.e "I saw a girl with a telescope."
     * Third one is constructed from the right of the sentence pair i.e "Donaudampfschifffahrts Kapitänsmützeninnenfuttersaum"
     */
    private Triple<MultiDataSet, MultiDataSet, MultiDataSet> generateMultiDataSets(Triple<Integer, Integer, Integer> maxLengths, boolean prependAppend) throws IOException {
        BertWordPieceTokenizerFactory t = new BertWordPieceTokenizerFactory(pathToVocab, false, false, c);
        int maxforPair = maxLengths.getFirst();
        int maxPartOne = maxLengths.getSecond();
        int maxPartTwo = maxLengths.getThird();
        BertIterator.Builder commonBuilder;
        commonBuilder = BertIterator.builder()
                .tokenizer(t)
                .minibatchSize(1)
                .featureArrays(BertIterator.FeatureArrays.INDICES_MASK_SEGMENTID)
                .vocabMap(t.getVocab())
                .task(BertIterator.Task.SEQ_CLASSIFICATION);
        BertIterator shortLongPairFirstIter = commonBuilder
                .lengthHandling(BertIterator.LengthHandling.FIXED_LENGTH, prependAppend ? maxforPair + 3 : maxforPair)
                .sentencePairProvider(new TestSentencePairProvider())
                .prependToken(prependAppend ? "[CLS]" : null)
                .appendToken(prependAppend ? "[SEP]" : null)
                .build();
        BertIterator shortFirstIter = commonBuilder
                .lengthHandling(BertIterator.LengthHandling.FIXED_LENGTH, prependAppend ? maxPartOne + 2 : maxPartOne)
                .sentenceProvider(new TestSentenceProvider())
                .prependToken(prependAppend ? "[CLS]" : null)
                .appendToken(prependAppend ? "[SEP]" : null)
                .build();
        BertIterator longFirstIter = commonBuilder
                .lengthHandling(BertIterator.LengthHandling.FIXED_LENGTH, prependAppend ? maxPartTwo + 1 : maxPartTwo)
                .sentenceProvider(new TestSentenceProvider(true))
                .prependToken(null)
                .appendToken(prependAppend ? "[SEP]" : null)
                .build();
        return new Triple<>(shortLongPairFirstIter.next(), shortFirstIter.next(), longFirstIter.next());
    }

    private static class TestSentenceProvider implements LabeledSentenceProvider {

        private int pos = 0;
        private boolean invert;

        private TestSentenceProvider() {
            this.invert = false;
        }

        private TestSentenceProvider(boolean invert) {
            this.invert = invert;
        }

        @Override
        public boolean hasNext() {
            return pos < totalNumSentences();
        }

        @Override
        public Pair<String, String> nextSentence() {
            Preconditions.checkState(hasNext());
            if (pos == 0) {
                pos++;
                if (!invert) return new Pair<>("I saw a girl with a telescope.", "positive");
                return new Pair<>("Donaudampfschifffahrts Kapitänsmützeninnenfuttersaum", "negative");
            } else {
                if (pos == 1) {
                    pos++;
                    if (!invert) return new Pair<>("Donaudampfschifffahrts Kapitänsmützeninnenfuttersaum", "negative");
                    return new Pair<>("I saw a girl with a telescope.", "positive");
                }
                pos++;
                if (!invert)
                    return new Pair<>("Goodnight noises everywhere", "positive");
                return new Pair<>("Goodnight moon", "positive");
            }
        }

        @Override
        public void reset() {
            pos = 0;
        }

        @Override
        public int totalNumSentences() {
            return 3;
        }

        @Override
        public List<String> allLabels() {
            return Arrays.asList("positive", "negative");
        }

        @Override
        public int numLabelClasses() {
            return 2;
        }
    }

    private static class TestSentencePairProvider implements LabeledPairSentenceProvider {

        private int pos = 0;

        @Override
        public boolean hasNext() {
            return pos < totalNumSentences();
        }

        @Override
        public Triple<String, String, String> nextSentencePair() {
            Preconditions.checkState(hasNext());
            if (pos == 0) {
                pos++;
                return new Triple<>("I saw a girl with a telescope.", "Donaudampfschifffahrts Kapitänsmützeninnenfuttersaum", "positive");
            } else {
                if (pos == 1) {
                    pos++;
                    return new Triple<>("Donaudampfschifffahrts Kapitänsmützeninnenfuttersaum", "I saw a girl with a telescope.", "negative");
                }
                pos++;
                return new Triple<>("Goodnight noises everywhere", "Goodnight moon", "positive");
            }
        }

        @Override
        public void reset() {
            pos = 0;
        }

        @Override
        public int totalNumSentences() {
            return 3;
        }

        @Override
        public List<String> allLabels() {
            return Arrays.asList("positive", "negative");
        }

        @Override
        public int numLabelClasses() {
            return 2;
        }
    }

}
