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
import org.deeplearning4j.iterator.bert.BertSequenceMasker;
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
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * BertIterator is a MultiDataSetIterator for training BERT (Transformer) models in the following way:<br>
 * (a) Unsupervised - Masked language model task (no sentence matching task is implemented thus far)<br>
 * (b) Supervised - For sequence classification (i.e., 1 label per sequence, typically used for fine tuning)<br>
 * The task can be specified using {@link Task}.
 * <br>
 * Example for unsupervised training:<br>
 * <pre>
 * {@code
 *
 * }
 * </pre>
 *
 * This iterator supports numerous ways of configuring the behaviour with respect to the sequence lengths and data layout.<br>
 * <br>
 * <u><b>{@link LengthHandling} configuration:</b></u><br>
 * Determines how to handle variable-length sequence situations.<br>
 * <b>FIXED_LENGTH</b>: Always trim longer sequences to the specified length, and always pad shorter sequences to the specified length.<br>
 * <b>ANY_LENGTH</b>: Output length is determined by the length of the longest sequence in the minibatch<br>
 * <b>CLIP_ONLY</b>: For any sequences longer than the specified maximum, clip them. If the maximum sequence length in
 * a minibatch is shorter than the specified maximum, no padding will occur.<br>
 *<br><br>
 * <u><b>{@link FeatureArrays} configuration:</b></u><br>
 * Determines what arrays should be included.
 * <b>INDICES_MASK</b>: Indices array and mask array only, no segment ID array<br>
 * <b>INDICES_MASK</b>: Indices array, mask array and segment ID array (which is all 0s for single segment tasks)<br>
 * <br>
 * <u><b>{@link UnsupervisedLabelFormat} configuration:</b></u><br>
 * Only relevant when the task is set to {@link Task#UNSUPERVISED}. Determine the format of the labels:<br>
 * <b>RANK2_IDX</b>: return int32 [minibatch, numTokens] array with entries being class numbers<br>
 * <b>RANK3_NCL</b>: return float32 [minibatch, numClasses, numTokens] array with 1-hot entries along dimension 1<br>
 * <b>RANK3_NLC</b>: return float32 [minibatch, numTokens, numClasses] array with 1-hot entries along dimension 2<br>
 * <br>
 */
public class BertIterator implements MultiDataSetIterator {

    public enum Task {UNSUPERVISED, SEQ_CLASSIFICATION}
    public enum LengthHandling {FIXED_LENGTH, ANY_LENGTH, CLIP_ONLY}
    public enum FeatureArrays {INDICES_MASK, INDICES_MASK_SEGMENTID}
    public enum UnsupervisedLabelFormat {RANK2_IDX, RANK3_NCL, RANK3_NLC}

    protected Task task;
    protected TokenizerFactory tokenizerFactory;
    protected int maxTokens = -1;
    protected int minibatchSize = 32;
    @Getter @Setter
    protected MultiDataSetPreProcessor preProcessor;
    protected LabeledSentenceProvider sentenceProvider = null;
    protected LengthHandling lengthHandling;
    protected FeatureArrays featureArrays;
    protected Map<String,Integer> vocabMap;   //TODO maybe use Eclipse ObjectIntHashMap or similar for fewer objects?
    protected BertSequenceMasker masker = null;
    protected UnsupervisedLabelFormat unsupervisedLabelFormat = null;
    protected String maskToken;


    protected List<String> vocabKeysAsList;

    protected BertIterator(Builder b){
        this.task = b.task;
        this.tokenizerFactory = b.tokenizerFactory;
        this.maxTokens = b.maxTokens;
        this.minibatchSize = b.minibatchSize;
        this.preProcessor = b.preProcessor;
        this.sentenceProvider = b.sentenceProvider;
        this.lengthHandling = b.lengthHandling;
        this.featureArrays = b.featureArrays;
        this.vocabMap = b.vocabMap;
        this.masker = b.masker;
        this.unsupervisedLabelFormat = b.unsupervisedLabelFormat;
        this.maskToken = b.maskToken;
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
                int idx = vocabMap.get(t.get(j));
                //TODO unknown token check
                outIdxs[i][j] = idx;
                outMask[i][j] = 1;
            }
        }

        INDArray outIdxsArr = Nd4j.createFromArray(outIdxs);
        INDArray outMaskArr = Nd4j.createFromArray(outMask);
        INDArray outSegmentIdArr = null;
        INDArray[] f;
        INDArray[] fm;
        if(featureArrays == FeatureArrays.INDICES_MASK_SEGMENTID){
            //For now: always segment index 0 (only single s sequence input supported)
            outSegmentIdArr = Nd4j.zeros(DataType.INT, mb, outLength);
            f = new INDArray[]{outIdxsArr, outSegmentIdArr};
            fm = new INDArray[]{outMaskArr, null};
        } else {
            f = new INDArray[]{outIdxsArr};
            fm = new INDArray[]{outMaskArr};
        }

        INDArray[] l = new INDArray[1];
        INDArray[] lm;
        if(task == Task.SEQ_CLASSIFICATION){
            int numClasses;
            int[] classLabels = new int[mb];
            if(sentenceProvider != null){
                numClasses = sentenceProvider.numLabelClasses();
                List<String> labels = sentenceProvider.allLabels();
                for(int i=0; i<mb; i++ ){
                    String lbl = tokenizedSentences.get(i).getRight();
                    classLabels[i] = labels.indexOf(lbl);
                    Preconditions.checkState(classLabels[i] >= 0, "Provided label \"%s\" for sentence does not exist in set of classes/categories", lbl);
                }
            } else {
                throw new RuntimeException();
            }
            l[0] = Nd4j.create(Nd4j.defaultFloatingPointType(), mb, numClasses);
            for( int i=0; i<mb; i++ ){
                l[0].putScalar(i, classLabels[i], 1.0);
            }
            lm = null;
        } else if(task == Task.UNSUPERVISED){
            if(vocabKeysAsList == null){
                String[] arr = new String[vocabMap.size()];
                for(Map.Entry<String,Integer> e : vocabMap.entrySet()){
                    arr[e.getValue()] = e.getKey();
                }
                vocabKeysAsList = Arrays.asList(arr);
            }


            int vocabSize = vocabMap.size();
            INDArray labelArr;
            INDArray mask = Nd4j.zeros(DataType.INT, mb, outLength);
            if(unsupervisedLabelFormat == UnsupervisedLabelFormat.RANK2_IDX){
                labelArr = Nd4j.create(DataType.INT, mb, outLength);
            } else if(unsupervisedLabelFormat == UnsupervisedLabelFormat.RANK3_NCL){
                labelArr = Nd4j.create(Nd4j.defaultFloatingPointType(), mb, vocabSize, outLength);
            } else if(unsupervisedLabelFormat == UnsupervisedLabelFormat.RANK3_NLC){
                labelArr = Nd4j.create(Nd4j.defaultFloatingPointType(), mb, outLength, vocabSize);
            } else {
                throw new IllegalStateException("Unknown unsupervised label format: " + unsupervisedLabelFormat);
            }

            for( int i=0; i<mb; i++ ){
                List<String> tokens = tokenizedSentences.get(i).getFirst();
                Pair<List<String>,boolean[]> p = masker.maskSequence(tokens, maskToken, vocabKeysAsList);
                List<String> maskedTokens = p.getFirst();
                boolean[] predictionTarget = p.getSecond();
                int seqLen = Math.min(predictionTarget.length, outLength);
                for(int j=0; j<seqLen; j++ ){
                    if(predictionTarget[j]){
                        String oldToken = tokenizedSentences.get(i).getFirst().get(j);  //This is target
                        int targetTokenIdx = vocabMap.get(oldToken);
                        if(unsupervisedLabelFormat == UnsupervisedLabelFormat.RANK2_IDX){
                            labelArr.putScalar(i, j, targetTokenIdx);
                        } else if(unsupervisedLabelFormat == UnsupervisedLabelFormat.RANK3_NCL){
                            labelArr.putScalar(i, j, targetTokenIdx, 1.0);
                        } else if(unsupervisedLabelFormat == UnsupervisedLabelFormat.RANK3_NLC){
                            labelArr.putScalar(i, targetTokenIdx, j, 1.0);
                        }

                        mask.putScalar(i, j, 1.0);

                        //Also update previously created feature label indexes:
                        String newToken = maskedTokens.get(j);
                        int newTokenIdx = vocabMap.get(newToken);
                        outIdxsArr.putScalar(i,j,newTokenIdx);
                    }
                }
            }
            l[0] = labelArr;
            lm = new INDArray[1];
            lm[0] = mask;
        } else {
            throw new IllegalStateException("Task not yet implemented: " + task);
        }

        org.nd4j.linalg.dataset.MultiDataSet mds = new org.nd4j.linalg.dataset.MultiDataSet(f, l, fm, lm);
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
        if(sentenceProvider != null){
            sentenceProvider.reset();
        }
    }

    public static Builder builder(){
        return new Builder();
    }

    public static class Builder {

        protected Task task;
        protected TokenizerFactory tokenizerFactory;
        protected LengthHandling lengthHandling = LengthHandling.FIXED_LENGTH;
        protected int maxTokens = -1;
        protected int minibatchSize = 32;
        protected MultiDataSetPreProcessor preProcessor;
        protected LabeledSentenceProvider sentenceProvider = null;
        protected FeatureArrays featureArrays = FeatureArrays.INDICES_MASK_SEGMENTID;
        protected Map<String,Integer> vocabMap;   //TODO maybe use Eclipse ObjectIntHashMap for fewer objects?
        protected BertSequenceMasker masker = null;
        protected UnsupervisedLabelFormat unsupervisedLabelFormat;
        protected String maskToken;

        public Builder task(Task task){
            this.task = task;
            return this;
        }

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

        public Builder outputArrays(FeatureArrays featureArrays){
            this.featureArrays = featureArrays;
            return this;
        }

        public Builder vocabMap(Map<String,Integer> vocabMap){
            this.vocabMap = vocabMap;
            return this;
        }

        public Builder masker(BertSequenceMasker masker){
            this.masker = masker;
            return this;
        }

        public Builder unsupervisedLabelFormat(UnsupervisedLabelFormat labelFormat){
            this.unsupervisedLabelFormat = labelFormat;
            return this;
        }

        public Builder maskToken(String maskToken){
            this.maskToken = maskToken;
            return this;
        }

        public BertIterator build(){
            Preconditions.checkState(task != null, "No task has been set. Use .task(BertIterator.Task.X) to set the task to be performed");
            Preconditions.checkState(tokenizerFactory != null, "No tokenizer factory has been set. A tokenizer factory (such as BertWordPieceTokenizerFactory) is required");
            Preconditions.checkState(vocabMap != null, "Cannot create iterator: No vocabMap has been set. Use Builder.vocabMap(Map<String,Integer>) to set");
            Preconditions.checkState(task != Task.UNSUPERVISED || masker != null, "If task is UNSUPERVISED training, a masker must be set via masker(BertSequenceMasker) method");
            Preconditions.checkState(task != Task.UNSUPERVISED || unsupervisedLabelFormat != null, "If task is UNSUPERVISED training, a label format must be set via masker(BertSequenceMasker) method");
            Preconditions.checkState(task != Task.UNSUPERVISED || maskToken != null, "If task is UNSUPERVISED training, the mask token in the vocab (such as \"[MASK]\" must be specified");

            return new BertIterator(this);
        }
    }

}
