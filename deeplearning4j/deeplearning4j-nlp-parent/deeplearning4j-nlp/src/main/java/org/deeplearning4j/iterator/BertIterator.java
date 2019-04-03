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
import org.deeplearning4j.iterator.bert.BertMaskedLMMasker;
import org.deeplearning4j.iterator.bert.BertSequenceMasker;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizerfactory.BertWordPieceTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
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
    protected boolean padMinibatches = false;
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
        this.padMinibatches = b.padMinibatches;
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
            throw new UnsupportedOperationException("Labelled sentence provider is null and no other iterator types have yet been implemented");
        }

        //Get and tokenize the sentences for this minibatch
        List<Pair<List<String>, String>> tokenizedSentences = new ArrayList<>(num);
        int longestSeq = -1;
        for(Pair<String,String> p : list){
            List<String> tokens = tokenizeSentence(p.getFirst());
            tokenizedSentences.add(new Pair<>(tokens, p.getSecond()));
            longestSeq = Math.max(longestSeq, tokens.size());
        }

        //Determine output array length...
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
        int mbPadded = padMinibatches ? minibatchSize : mb;
        int[][] outIdxs = new int[mbPadded][outLength];
        int[][] outMask = new int[mbPadded][outLength];

        for( int i=0; i<tokenizedSentences.size(); i++ ){
            Pair<List<String>,String> p = tokenizedSentences.get(i);
            List<String> t = p.getFirst();
            for( int j=0; j<outLength && j<t.size(); j++ ){
                Preconditions.checkState(vocabMap.containsKey(t.get(j)), "Unknown token encontered: token \"%s\" is not in vocabulary", t.get(j));
                int idx = vocabMap.get(t.get(j));
                outIdxs[i][j] = idx;
                outMask[i][j] = 1;
            }
        }

        //Create actual arrays. Indices, mask, and optional segment ID
        INDArray outIdxsArr = Nd4j.createFromArray(outIdxs);
        INDArray outMaskArr = Nd4j.createFromArray(outMask);
        INDArray outSegmentIdArr;
        INDArray[] f;
        INDArray[] fm;
        if(featureArrays == FeatureArrays.INDICES_MASK_SEGMENTID){
            //For now: always segment index 0 (only single s sequence input supported)
            outSegmentIdArr = Nd4j.zeros(DataType.INT, mbPadded, outLength);
            f = new INDArray[]{outIdxsArr, outSegmentIdArr};
            fm = new INDArray[]{outMaskArr, null};
        } else {
            f = new INDArray[]{outIdxsArr};
            fm = new INDArray[]{outMaskArr};
        }

        INDArray[] l = new INDArray[1];
        INDArray[] lm;
        if(task == Task.SEQ_CLASSIFICATION){
            //Sequence classification task: output is 2d, one-hot, shape [minibatch, numClasses]
            int numClasses;
            int[] classLabels = new int[mbPadded];
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
            l[0] = Nd4j.create(Nd4j.defaultFloatingPointType(), mbPadded, numClasses);
            for( int i=0; i<mb; i++ ){
                l[0].putScalar(i, classLabels[i], 1.0);
            }
            lm = null;
            if(padMinibatches && mb != mbPadded){
                INDArray a = Nd4j.zeros(DataType.FLOAT, mbPadded, 1);
                lm = new INDArray[]{a};
                a.get(NDArrayIndex.interval(0, mb), NDArrayIndex.all()).assign(1);
            }
        } else if(task == Task.UNSUPERVISED){
            //Unsupervised, masked language model task
            //Output is either 2d, or 3d depending on settings
            if(vocabKeysAsList == null){
                String[] arr = new String[vocabMap.size()];
                for(Map.Entry<String,Integer> e : vocabMap.entrySet()){
                    arr[e.getValue()] = e.getKey();
                }
                vocabKeysAsList = Arrays.asList(arr);
            }


            int vocabSize = vocabMap.size();
            INDArray labelArr;
            INDArray lMask = Nd4j.zeros(DataType.INT, mbPadded, outLength);
            if(unsupervisedLabelFormat == UnsupervisedLabelFormat.RANK2_IDX){
                labelArr = Nd4j.create(DataType.INT, mbPadded, outLength);
            } else if(unsupervisedLabelFormat == UnsupervisedLabelFormat.RANK3_NCL){
                labelArr = Nd4j.create(Nd4j.defaultFloatingPointType(), mbPadded, vocabSize, outLength);
            } else if(unsupervisedLabelFormat == UnsupervisedLabelFormat.RANK3_NLC){
                labelArr = Nd4j.create(Nd4j.defaultFloatingPointType(), mbPadded, outLength, vocabSize);
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

                        lMask.putScalar(i, j, 1.0);

                        //Also update previously created feature label indexes:
                        String newToken = maskedTokens.get(j);
                        int newTokenIdx = vocabMap.get(newToken);
                        outIdxsArr.putScalar(i,j,newTokenIdx);
                    }
                }
            }
            l[0] = labelArr;
            lm = new INDArray[1];
            lm[0] = lMask;
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
        protected boolean padMinibatches = false;
        protected MultiDataSetPreProcessor preProcessor;
        protected LabeledSentenceProvider sentenceProvider = null;
        protected FeatureArrays featureArrays = FeatureArrays.INDICES_MASK_SEGMENTID;
        protected Map<String,Integer> vocabMap;   //TODO maybe use Eclipse ObjectIntHashMap for fewer objects?
        protected BertSequenceMasker masker = new BertMaskedLMMasker();
        protected UnsupervisedLabelFormat unsupervisedLabelFormat;
        protected String maskToken;

        /**
         * Specify the {@link Task} the iterator should be set up for. See {@link BertIterator} for more details.
         */
        public Builder task(Task task){
            this.task = task;
            return this;
        }

        /**
         * Specify the TokenizerFactory to use.
         * For BERT, typically {@link org.deeplearning4j.text.tokenization.tokenizerfactory.BertWordPieceTokenizerFactory}
         * is used
         */
        public Builder tokenizer(TokenizerFactory tokenizerFactory){
            this.tokenizerFactory = tokenizerFactory;
            return this;
        }

        /**
         * Specifies how the sequence length of the output data should be handled. See {@link BertIterator} for more details.
         * @param lengthHandling    Length handling
         * @param maxLength         Not used if LengthHandling is set to {@link LengthHandling#ANY_LENGTH}
         * @return
         */
        public Builder lengthHandling(@NonNull LengthHandling lengthHandling, int maxLength){
            this.lengthHandling = lengthHandling;
            this.maxTokens = maxLength;
            return this;
        }

        /**
         * Minibatch size to use (number of examples to train on for each iteration)
         * See also: {@link #padMinibatches}
         * @param minibatchSize    Minibatch size
         */
        public Builder minibatchSize(int minibatchSize){
            this.minibatchSize = minibatchSize;
            return this;
        }

        /**
         * Default: false (disabled)<br>
         * If the dataset is not an exact multiple of the minibatch size, should we pad the smaller final minibatch?<br>
         * For example, if we have 100 examples total, and 32 minibatch size, the following number of examples will be returned
         * for subsequent calls of next() in the one epoch:<br>
         * padMinibatches = false (default): 32, 32, 32, 4.<br>
         * padMinibatches = true: 32, 32, 32, 32 (note: the last minibatch will have 4 real examples, and 28 masked out padding examples).<br>
         * Both options should result in exactly the same model. However, some BERT implementations may require exactly an
         * exact number of examples in all minibatches to function.
         */
        public Builder padMinibatches(boolean padMinibatches){
            this.padMinibatches = padMinibatches;
            return this;
        }

        /**
         * Set the preprocessor to be used on the MultiDataSets before returning them. Default: none (null)
         */
        public Builder preProcessor(MultiDataSetPreProcessor preProcessor){
            this.preProcessor = preProcessor;
            return this;
        }

        /**
         * Specify the source of the data for classification. Can also be used for unsupervised learning; in the unsupervised
         * use case, the labels will be ignored.
         */
        public Builder sentenceProvider(LabeledSentenceProvider sentenceProvider){
            this.sentenceProvider = sentenceProvider;
            return this;
        }

        /**
         * Specify what arrays should be returned. See {@link BertIterator} for more details.
         */
        public Builder featureArrays(FeatureArrays featureArrays){
            this.featureArrays = featureArrays;
            return this;
        }

        /**
         * Provide the vocabulary as a map. Keys are
         * If using {@link BertWordPieceTokenizerFactory},
         * this can be obtained using {@link BertWordPieceTokenizerFactory#getVocab()}
         */
        public Builder vocabMap(Map<String,Integer> vocabMap){
            this.vocabMap = vocabMap;
            return this;
        }

        /**
         * Used only for unsupervised training (i.e., when task is set to {@link Task#UNSUPERVISED} for learning a
         * masked language model. This can be used to customize how the masking is performed.<br>
         * Default: {@link BertMaskedLMMasker}
         */
        public Builder masker(BertSequenceMasker masker){
            this.masker = masker;
            return this;
        }

        /**
         * Used only for unsupervised training (i.e., when task is set to {@link Task#UNSUPERVISED} for learning a
         * masked language model. Used to specify the format that the labels should be returned in.
         * See {@link BertIterator} for more details.
         */
        public Builder unsupervisedLabelFormat(UnsupervisedLabelFormat labelFormat){
            this.unsupervisedLabelFormat = labelFormat;
            return this;
        }

        /**
         * Used only for unsupervised training (i.e., when task is set to {@link Task#UNSUPERVISED} for learning a
         * masked language model. This specifies the token (such as "[MASK]") that should be used when a value is masked out.
         * Note that this is passed to the {@link BertSequenceMasker} defined by {@link #masker(BertSequenceMasker)} hence
         * the exact behaviour will depend on what masker is used.<br>
         * Note that this must be in the vocabulary map set in {@link #vocabMap}
         */
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
