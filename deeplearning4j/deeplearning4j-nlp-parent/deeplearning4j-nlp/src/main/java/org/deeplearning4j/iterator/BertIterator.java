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
import lombok.NonNull;
import lombok.Setter;
import org.deeplearning4j.text.tokenization.tokenizer.BertWordPieceStreamTokenizer;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class BertIterator implements MultiDataSetIterator {

    public enum Task {UNSUPERVISED, SEQ_CLASSIFICATION, CLASSIFICATION}
    public enum LengthHandling {FIXED_LENGTH, ANY_LENGTH, CLIP_ONLY}
    public enum OutputArrays {INDICES_MASK, INDICES_MASK_SEGMENTID}

    protected TokenizerFactory tokenizerFactory;
    protected int maxTokens = -1;
    protected int minibatchSize = 32;
    @Getter @Setter
    protected MultiDataSetPreProcessor preProcessor;
    protected LabeledSentenceProvider sentenceProvider = null;
    protected LengthHandling lengthHandling;
    protected OutputArrays outputArrays;
    protected Map<String,Integer> tokens;   //TODO maybe use Eclipse ObjectIntHashMap for fewer objects?

    protected BertIterator(Builder b){
        this.tokenizerFactory = b.tokenizerFactory;
        this.maxTokens = b.maxTokens;
        this.minibatchSize = b.minibatchSize;
        this.preProcessor = b.preProcessor;
        this.sentenceProvider = b.sentenceProvider;
        this.lengthHandling = b.lengthHandling;
        this.outputArrays = b.outputArrays;
        this.tokens = b.tokens;
    }

    @Override
    public boolean hasNext() {
        return sentenceProvider.hasNext();
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
        int longestSeq = -1;
        for(Pair<String,String> p : list){
            List<String> tokens = tokenizeSentence(p.getFirst());
            tokenizedSentences.add(new Pair<>(tokens, p.getSecond()));
            longestSeq = Math.max(longestSeq, tokens.size());
        }

        int outLength;
        switch (lengthHandling){
            case FIXED_LENGTH:
                outLength = maxTokens;
                break;
            case ANY_LENGTH:
                outLength = longestSeq;
                break;
            case CLIP_ONLY:
                outLength = Math.min(maxTokens, longestSeq);
                break;
            default:
                throw new RuntimeException("Not implemented length handling mode: " + lengthHandling);
        }

        int mb = tokenizedSentences.size();

        int[][] outIdxs = new int[mb][outLength];
        int[][] outMask = new int[mb][outLength];

        for( int i=0; i<tokenizedSentences.size(); i++ ){
            Pair<List<String>,String> p = tokenizedSentences.get(i);
            List<String> t = p.getFirst();
            for( int j=0; j<outLength && j<t.size(); j++ ){
                int idx = tokens.get(t.get(j));
                //TODO unknown token check
                outIdxs[i][j] = idx;
                outMask[i][j] = 1;
            }
        }

        INDArray outIdxsArr = Nd4j.createFromArray(outIdxs);
        INDArray outMaskArr = Nd4j.createFromArray(outMask);
        INDArray outSegmentIdArr = null;
        INDArray[] f;
        if(outputArrays == OutputArrays.INDICES_MASK_SEGMENTID){
            //For now: always segment index 0 (only single s sequence input supported)
            outSegmentIdArr = Nd4j.zeros(DataType.INT, mb, outLength);
            f = new INDArray[]{outIdxsArr, outMaskArr, outSegmentIdArr};
        } else {
            f = new INDArray[]{outIdxsArr, outMaskArr};
        }

        INDArray[] l = null; //TODO

        org.nd4j.linalg.dataset.MultiDataSet mds = new org.nd4j.linalg.dataset.MultiDataSet(f, l);
        if(preProcessor != null)
            preProcessor.preProcess(mds);

        return mds;
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
        protected LengthHandling lengthHandling = LengthHandling.FIXED_LENGTH;
        protected int maxTokens = -1;
        protected int minibatchSize = 32;
        protected MultiDataSetPreProcessor preProcessor;
        protected LabeledSentenceProvider sentenceProvider = null;
        protected OutputArrays outputArrays = OutputArrays.INDICES_MASK_SEGMENTID;
        protected Map<String,Integer> tokens;   //TODO maybe use Eclipse ObjectIntHashMap for fewer objects?

        public Builder tokenizer(TokenizerFactory tokenizerFactory){
            this.tokenizerFactory = tokenizerFactory;
            return this;
        }

        public Builder lengthHandling(@NonNull LengthHandling lengthHandling, int maxLength){
            this.lengthHandling = lengthHandling;
            this.maxTokens = maxLength;
            return this;
        }

        public Builder minibatchSize(int minibatchSize){
            this.minibatchSize = minibatchSize;
            return this;
        }

        public Builder preProcessor(MultiDataSetPreProcessor preProcessor){
            this.preProcessor = preProcessor;
            return this;
        }

        public Builder sentenceProvider(LabeledSentenceProvider sentenceProvider){
            this.sentenceProvider = sentenceProvider;
            return this;
        }

        public Builder outputArrays(OutputArrays outputArrays){
            this.outputArrays = outputArrays;
            return this;
        }

        public Builder tokens(Map<String,Integer> tokens){
            this.tokens = tokens;
            return this;
        }

        public BertIterator build(){
            Preconditions.checkState(tokenizerFactory != null, "No tokenizer factory has been set. A tokenizer factory (such as BertWordPieceTokenizerFactory) is required");
            Preconditions.checkState(tokens != null, "Cannot create iterator: No tokens have bees net. Use Builder.tokens(Map<String,Integer>) to set");



            return new BertIterator(this);
        }
    }

}
