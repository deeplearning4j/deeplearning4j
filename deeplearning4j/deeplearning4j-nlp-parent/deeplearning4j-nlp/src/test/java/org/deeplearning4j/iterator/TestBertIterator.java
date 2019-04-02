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

import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.LowCasePreProcessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.BertWordPieceTokenizerFactory;
import org.junit.Test;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.primitives.Pair;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;


public class TestBertIterator {

    private File pathToVocab =  new ClassPathResource("other/vocab.txt").getFile();

    public TestBertIterator() throws IOException{ }

    @Test
    public void testBertIterator() throws Exception {

        String toTokenize = "I saw a girl with a telescope.";
        BertWordPieceTokenizerFactory t = new BertWordPieceTokenizerFactory(pathToVocab);
        t.setTokenPreProcessor(new LowCasePreProcessor());

        BertIterator b = BertIterator.builder()
                .tokenizer(t)
                .lengthHandling(BertIterator.LengthHandling.FIXED_LENGTH, 16)
                .minibatchSize(2)
                .sentenceProvider(new TestSentenceProvider())
                .outputArrays(BertIterator.OutputArrays.INDICES_MASK)
                .tokens(t.getVocab())
                .build();

        MultiDataSet mds = b.next();
        assertEquals(2, mds.getFeatures().length);
        System.out.println(mds.getFeatures(0));
        System.out.println(mds.getFeatures(1));

    }

    private static class TestSentenceProvider implements LabeledSentenceProvider {

        private int pos = 0;

        @Override
        public boolean hasNext() {
            return pos == 0;
        }

        @Override
        public Pair<String, String> nextSentence() {
            Preconditions.checkState(hasNext());
            pos++;
            return new Pair<>("I saw a girl with a telescope.", "positive");
        }

        @Override
        public void reset() {
            pos = 0;
        }

        @Override
        public int totalNumSentences() {
            return 1;
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
