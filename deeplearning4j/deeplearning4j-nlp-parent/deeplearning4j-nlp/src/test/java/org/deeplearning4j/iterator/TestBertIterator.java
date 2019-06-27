/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.resources.Resources;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;

import static org.junit.Assert.*;


public class TestBertIterator extends BaseDL4JTest {

    private File pathToVocab = Resources.asFile("other/vocab.txt");
    private static Charset c = StandardCharsets.UTF_8;

    public TestBertIterator() throws IOException{ }

    @Test(timeout = 20000L)
    public void testBertSequenceClassification() throws Exception {

        String toTokenize1 = "I saw a girl with a telescope.";
        String toTokenize2 = "Donaudampfschifffahrts Kapitänsmützeninnenfuttersaum";
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
        Map<String,Integer> m = t.getVocab();
        for( int i=0; i<tokens.size(); i++ ){
            int idx = m.get(tokens.get(i));
            expEx0.putScalar(0, i, idx);
            expM0.putScalar(0, i, 1);
        }

        INDArray expEx1 = Nd4j.create(DataType.INT, 1, 16);
        INDArray expM1 = Nd4j.create(DataType.INT, 1, 16);
        List<String> tokens2 = t.create(toTokenize2).getTokens();
        for( int i=0; i<tokens2.size(); i++ ){
            String token = tokens2.get(i);
            if(!m.containsKey(token)){
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

        assertFalse(b.hasNext());
        b.reset();
        assertTrue(b.hasNext());
        MultiDataSet mds2 = b.next();

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
        //Segment ID should be all 0s for single segment task
        INDArray segmentId = expM.like();
        assertEquals(segmentId, mds.getFeatures(1));
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

        assertFalse(b.hasNext());
        b.reset();
        mds = b.next();
    }

    @Test(timeout = 20000L)
    public void testLengthHandling() throws Exception {
        String toTokenize1 = "I saw a girl with a telescope.";
        String toTokenize2 = "Donaudampfschifffahrts Kapitänsmützeninnenfuttersaum";
        BertWordPieceTokenizerFactory t = new BertWordPieceTokenizerFactory(pathToVocab, false, false, c);
        INDArray expEx0 = Nd4j.create(DataType.INT, 1, 16);
        INDArray expM0 = Nd4j.create(DataType.INT, 1, 16);
        List<String> tokens = t.create(toTokenize1).getTokens();
        Map<String,Integer> m = t.getVocab();
        for( int i=0; i<tokens.size(); i++ ){
            int idx = m.get(tokens.get(i));
            expEx0.putScalar(0, i, idx);
            expM0.putScalar(0, i, 1);
        }

        INDArray expEx1 = Nd4j.create(DataType.INT, 1, 16);
        INDArray expM1 = Nd4j.create(DataType.INT, 1, 16);
        List<String> tokens2 = t.create(toTokenize2).getTokens();
        for( int i=0; i<tokens2.size(); i++ ){
            String token = tokens2.get(i);
            if(!m.containsKey(token)){
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
        assertEquals(expF.get(NDArrayIndex.all(), NDArrayIndex.interval(0,14)), mds.getFeatures(0));
        assertEquals(expM.get(NDArrayIndex.all(), NDArrayIndex.interval(0,14)), mds.getFeaturesMaskArray(0));

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
        BertWordPieceTokenizerFactory t = new BertWordPieceTokenizerFactory(pathToVocab, false, false, c);
        INDArray expEx0 = Nd4j.create(DataType.INT, 1, 16);
        INDArray expM0 = Nd4j.create(DataType.INT, 1, 16);
        List<String> tokens = t.create(toTokenize1).getTokens();
        Map<String,Integer> m = t.getVocab();
        for( int i=0; i<tokens.size(); i++ ){
            int idx = m.get(tokens.get(i));
            expEx0.putScalar(0, i, idx);
            expM0.putScalar(0, i, 1);
        }

        INDArray expEx1 = Nd4j.create(DataType.INT, 1, 16);
        INDArray expM1 = Nd4j.create(DataType.INT, 1, 16);
        List<String> tokens2 = t.create(toTokenize2).getTokens();
        for( int i=0; i<tokens2.size(); i++ ){
            String token = tokens2.get(i);
            if(!m.containsKey(token)){
                throw new IllegalStateException("Unknown token: \"" + token + "\"");
            }
            int idx = m.get(token);
            expEx1.putScalar(0, i, idx);
            expM1.putScalar(0, i, 1);
        }

        INDArray zeros = Nd4j.create(DataType.INT, 2, 16);

        INDArray expF = Nd4j.vstack(expEx0, expEx1, zeros);
        INDArray expM = Nd4j.vstack(expM0, expM1, zeros);
        INDArray expL = Nd4j.createFromArray(new float[][]{{1, 0}, {0, 1}, {0, 0}, {0, 0}});
        INDArray expLM = Nd4j.create(DataType.FLOAT, 4, 1);
        expLM.putScalar(0, 0, 1);
        expLM.putScalar(1, 0, 1);

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
    }

    private static class TestSentenceProvider implements LabeledSentenceProvider {

        private int pos = 0;

        @Override
        public boolean hasNext() {
            return pos < totalNumSentences();
        }

        @Override
        public Pair<String, String> nextSentence() {
            Preconditions.checkState(hasNext());
            if(pos++ == 0){
                return new Pair<>("I saw a girl with a telescope.", "positive");
            } else {
                return new Pair<>("Donaudampfschifffahrts Kapitänsmützeninnenfuttersaum", "negative");
            }
        }

        @Override
        public void reset() {
            pos = 0;
        }

        @Override
        public int totalNumSentences() {
            return 2;
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
