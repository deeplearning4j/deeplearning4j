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

import lombok.Getter;
import lombok.Setter;
import org.deeplearning4j.text.tokenization.tokenizer.BertWordPieceStreamTokenizer;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.primitives.Pair;

import java.util.ArrayList;
import java.util.List;

public class BertIterator implements MultiDataSetIterator {

    public enum Task {UNSUPERVISED, SEQ_CLASSIFICATION, CLASSIFICATION}
    public enum LengthHandling {FIXED_LENGTH, ANY_LENGTH, CLIP_ONLY}

    protected TokenizerFactory tokenizerFactory;
    protected int maxTokens = -1;
    protected int minibatchSize = 32;
    @Getter @Setter
    protected MultiDataSetPreProcessor preProcessor;
    protected LabeledSentenceProvider sentenceProvider = null;

    protected BertIterator(Builder builder){

    }

    @Override
    public boolean hasNext() {
        return false;
    }

    @Override
    public MultiDataSet next() {
        return next(minibatchSize);
    }

    @Override
    public void remove() {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public MultiDataSet next(int num) {
        Preconditions.checkState(hasNext(), "No next element available");

        List<Pair<String,String>> list = new ArrayList<>(num);
        int count = 0;
        if(sentenceProvider != null){
            while(sentenceProvider.hasNext() && count++ < num) {
                list.add(sentenceProvider.nextSentence());
            }
        } else {
            //TODO - other types of iterators...
        }

        List<Pair<List<String>, String>> tokenizedSentences = new ArrayList<>(num);
        int maxLength = -1;
        for(Pair<String,String> p : list){
            List<String> tokens = tokenizeSentence(p.getFirst());
            tokenizedSentences.add(new Pair<>(tokens, p.getSecond()));
            maxLength = Math.max(maxLength, tokens.size());
        }

        

        return null;
    }

    private List<String> tokenizeSentence(String sentence) {
        Tokenizer t = tokenizerFactory.create(sentence);

        List<String> tokens = new ArrayList<>();
        while (t.hasMoreTokens()) {
            String token = t.nextToken();
            tokens.add(token);
        }
        return tokens;
    }

    @Override
    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return true;
    }

    @Override
    public void reset() {

    }

    public static Builder builder(){
        return new Builder();
    }

    public static class Builder {


        protected TokenizerFactory tokenizerFactory;
        protected int maxTokens = -1;
        protected int minibatchSize = 32;
        protected MultiDataSetPreProcessor preProcessor;
        protected LabeledSentenceProvider sentenceProvider = null;


        public BertIterator build(){
            Preconditions.checkState(tokenizerFactory != null, "No tokenizer factory has been set. A tokenizer factory (such as BertWordPieceTokenizerFactory) is required");


        }
    }

}
