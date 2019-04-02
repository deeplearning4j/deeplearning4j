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

import org.deeplearning4j.iterator.bert.BertMaskedLMMasker;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.LowCasePreProcessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.BertWordPieceTokenizerFactory;
import org.junit.Test;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.primitives.Pair;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;

import static org.junit.Assert.assertEquals;


public class TestBertIterator {

    private File pathToVocab =  new ClassPathResource("other/vocab.txt").getFile();

    public TestBertIterator() throws IOException{ }

    @Test
    public void testBertIteratorBasic() throws Exception {

        String toTokenize1 = "I saw a girl with a telescope.";
        String toTokenize2 = "Donaudampfschifffahrts Kapitänsmützeninnenfuttersaum";
        BertWordPieceTokenizerFactory t = new BertWordPieceTokenizerFactory(pathToVocab);

        BertIterator b = BertIterator.builder()
                .tokenizer(t)
                .lengthHandling(BertIterator.LengthHandling.FIXED_LENGTH, 16)
                .minibatchSize(2)
                .sentenceProvider(new TestSentenceProvider())
                .outputArrays(BertIterator.OutputArrays.INDICES_MASK)
                .vocabMap(t.getVocab())
                .task(BertIterator.Task.SEQ_CLASSIFICATION)
                .build();

        MultiDataSet mds = b.next();
        assertEquals(2, mds.getFeatures().length);
        System.out.println(mds.getFeatures(0));
        System.out.println(mds.getFeatures(1));


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
        assertEquals(expM, mds.getFeatures(1));
    }

    @Test
    public void testBertTasks() throws Exception {
        //Task 1: Unsupervised
        BertWordPieceTokenizerFactory t = new BertWordPieceTokenizerFactory(pathToVocab);
        BertIterator b = BertIterator.builder()
                .tokenizer(t)
                .lengthHandling(BertIterator.LengthHandling.FIXED_LENGTH, 16)
                .minibatchSize(2)
                .sentenceProvider(new TestSentenceProvider())
                .outputArrays(BertIterator.OutputArrays.INDICES_MASK)
                .vocabMap(t.getVocab())
                .task(BertIterator.Task.UNSUPERVISED)
                .masker(new BertMaskedLMMasker(new Random(12345), 0.3, 0.5, 0.5))
                .unsupervisedLabelFormat(BertIterator.UnsupervisedLabelFormat.RANK2_IDX)
                .maskToken("[MASK]")
                .build();

//        for(String s : t.getVocab().keySet()){
//            if(s.contains("mask") || s.contains("[")){
//                System.out.println(s);
//            }
//        }

        MultiDataSet mds = b.next();
        System.out.println(mds.getFeatures(0));
        System.out.println(mds.getFeatures(1));
        System.out.println(mds.getLabels(0));


        //Task 2: Sequence classification



        //Task 3: Per token classification
    }

    private static class TestSentenceProvider implements LabeledSentenceProvider {

        private int pos = 0;

        @Override
        public boolean hasNext() {
            return pos <= totalNumSentences();
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
