/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
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
import lombok.NonNull;
import lombok.Setter;
import org.deeplearning4j.iterator.bert.BertMaskedLMMasker;
import org.deeplearning4j.iterator.bert.BertSequenceMasker;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizerfactory.BertWordPieceTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.common.primitives.Pair;
import org.nd4j.common.primitives.Triple;

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
 * <b>Example for unsupervised training:</b><br>
 * <pre>
 * {@code
 *          BertWordPieceTokenizerFactory t = new BertWordPieceTokenizerFactory(pathToVocab);
 *          BertIterator b = BertIterator.builder()
 *              .tokenizer(t)
 *              .lengthHandling(BertIterator.LengthHandling.FIXED_LENGTH, 16)
 *              .minibatchSize(2)
 *              .sentenceProvider(<sentence provider here>)
 *              .featureArrays(BertIterator.FeatureArrays.INDICES_MASK)
 *              .vocabMap(t.getVocab())
 *              .task(BertIterator.Task.UNSUPERVISED)
 *              .masker(new BertMaskedLMMasker(new Random(12345), 0.2, 0.5, 0.5))
 *              .unsupervisedLabelFormat(BertIterator.UnsupervisedLabelFormat.RANK2_IDX)
 *              .maskToken("[MASK]")
 *              .build();
 * }
 * </pre>
 * <br>
 * <b>Example for supervised (sequence classification - one label per sequence) training:</b><br>
 * <pre>
 * {@code
 *          BertWordPieceTokenizerFactory t = new BertWordPieceTokenizerFactory(pathToVocab);
 *          BertIterator b = BertIterator.builder()
 *              .tokenizer(t)
 *              .lengthHandling(BertIterator.LengthHandling.FIXED_LENGTH, 16)
 *              .minibatchSize(2)
 *              .sentenceProvider(new TestSentenceProvider())
 *              .featureArrays(BertIterator.FeatureArrays.INDICES_MASK)
 *              .vocabMap(t.getVocab())
 *              .task(BertIterator.Task.SEQ_CLASSIFICATION)
 *              .build();
 * }
 * </pre>
 * <br>
 * <b>Example to use an instantiated iterator for inference:</b><br>
 * <pre>
 * {@code
 *          BertIterator b;
 *          Pair<INDArray[],INDArray[]> featuresAndMask;
 *          INDArray[] features;
 *          INDArray[] featureMasks;
 *
 *          //With sentences
 *          List<String> forInference;
 *          featuresAndMask = b.featurizeSentences(forInference);
 *
 *          //OR with sentence pairs
 *          List<Pair<String, String>> forInferencePair};
 *          featuresAndMask = b.featurizeSentencePairs(forInference);
 *
 *          features = featuresAndMask.getFirst();
 *          featureMasks = featuresAndMask.getSecond();
 * }
 * </pre>
 * This iterator supports numerous ways of configuring the behaviour with respect to the sequence lengths and data layout.<br>
 * <br>
 * <u><b>{@link LengthHandling} configuration:</b></u><br>
 * Determines how to handle variable-length sequence situations.<br>
 * <b>FIXED_LENGTH</b>: Always trim longer sequences to the specified length, and always pad shorter sequences to the specified length.<br>
 * <b>ANY_LENGTH</b>: Output length is determined by the length of the longest sequence in the minibatch. Shorter sequences within the
 * minibatch are zero padded and masked.<br>
 * <b>CLIP_ONLY</b>: For any sequences longer than the specified maximum, clip them. If the maximum sequence length in
 * a minibatch is shorter than the specified maximum, no padding will occur. For sequences that are shorter than the
 * maximum (within the current minibatch) they will be zero padded and masked.<br>
 * <br><br>
 * <u><b>{@link FeatureArrays} configuration:</b></u><br>
 * Determines what arrays should be included.<br>
 * <b>INDICES_MASK</b>: Indices array and mask array only, no segment ID array. Returns 1 feature array, 1 feature mask array (plus labels).<br>
 * <b>INDICES_MASK_SEGMENTID</b>: Indices array, mask array and segment ID array (which is all 0s for single segment tasks). Returns
 * 2 feature arrays (indices, segment ID) and 1 feature mask array (plus labels)<br>
 * <br>
 * <u><b>{@link UnsupervisedLabelFormat} configuration:</b></u><br>
 * Only relevant when the task is set to {@link Task#UNSUPERVISED}. Determine the format of the labels:<br>
 * <b>RANK2_IDX</b>: return int32 [minibatch, numTokens] array with entries being class numbers. Example use case: with sparse softmax loss functions.<br>
 * <b>RANK3_NCL</b>: return float32 [minibatch, numClasses, numTokens] array with 1-hot entries along dimension 1. Example use case: RnnOutputLayer, RnnLossLayer<br>
 * <b>RANK3_LNC</b>: return float32 [numTokens, minibatch, numClasses] array with 1-hot entries along dimension 2. This format is occasionally
 * used for some RNN layers in libraries such as TensorFlow, for example<br>
 * <br>
 */
public class BertIterator implements MultiDataSetIterator {

    public enum Task {UNSUPERVISED, SEQ_CLASSIFICATION}

    public enum LengthHandling {FIXED_LENGTH, ANY_LENGTH, CLIP_ONLY}

    public enum FeatureArrays {INDICES_MASK, INDICES_MASK_SEGMENTID}

    public enum UnsupervisedLabelFormat {RANK2_IDX, RANK3_NCL, RANK3_LNC}

    protected Task task;
    protected TokenizerFactory tokenizerFactory;
    protected int maxTokens = -1;
    protected int minibatchSize = 32;
    protected boolean padMinibatches = false;
    @Getter
    @Setter
    protected MultiDataSetPreProcessor preProcessor;
    protected LabeledSentenceProvider sentenceProvider = null;
    protected LabeledPairSentenceProvider sentencePairProvider = null;
    protected LengthHandling lengthHandling;
    protected FeatureArrays featureArrays;
    protected Map<String, Integer> vocabMap;   //TODO maybe use Eclipse ObjectIntHashMap or similar for fewer objects?
    protected BertSequenceMasker masker = null;
    protected UnsupervisedLabelFormat unsupervisedLabelFormat = null;
    protected String maskToken;
    protected String prependToken;
    protected String appendToken;


    protected List<String> vocabKeysAsList;

    protected BertIterator(Builder b) {
        this.task = b.task;
        this.tokenizerFactory = b.tokenizerFactory;
        this.maxTokens = b.maxTokens;
        this.minibatchSize = b.minibatchSize;
        this.padMinibatches = b.padMinibatches;
        this.preProcessor = b.preProcessor;
        this.sentenceProvider = b.sentenceProvider;
        this.sentencePairProvider = b.sentencePairProvider;
        this.lengthHandling = b.lengthHandling;
        this.featureArrays = b.featureArrays;
        this.vocabMap = b.vocabMap;
        this.masker = b.masker;
        this.unsupervisedLabelFormat = b.unsupervisedLabelFormat;
        this.maskToken = b.maskToken;
        this.prependToken = b.prependToken;
        this.appendToken = b.appendToken;
    }

    @Override
    public boolean hasNext() {
        if (sentenceProvider != null)
            return sentenceProvider.hasNext();
        return sentencePairProvider.hasNext();
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
        List<Pair<List<String>, String>> tokensAndLabelList;
        int mbSize = 0;
        int outLength;
        long[] segIdOnesFrom = null;
        if (sentenceProvider != null) {
            List<Pair<String, String>> list = new ArrayList<>(num);
            while (sentenceProvider.hasNext() && mbSize++ < num) {
                list.add(sentenceProvider.nextSentence());
            }
            SentenceListProcessed sentenceListProcessed = tokenizeMiniBatch(list);
            tokensAndLabelList = sentenceListProcessed.getTokensAndLabelList();
            outLength = sentenceListProcessed.getMaxL();
        } else if (sentencePairProvider != null) {
            List<Triple<String, String, String>> listPairs = new ArrayList<>(num);
            while (sentencePairProvider.hasNext() && mbSize++ < num) {
                listPairs.add(sentencePairProvider.nextSentencePair());
            }
            SentencePairListProcessed sentencePairListProcessed = tokenizePairsMiniBatch(listPairs);
            tokensAndLabelList = sentencePairListProcessed.getTokensAndLabelList();
            outLength = sentencePairListProcessed.getMaxL();
            segIdOnesFrom = sentencePairListProcessed.getSegIdOnesFrom();
        } else {
            //TODO - other types of iterators...
            throw new UnsupportedOperationException("Labelled sentence provider is null and no other iterator types have yet been implemented");
        }

        Pair<INDArray[], INDArray[]> featuresAndMaskArraysPair = convertMiniBatchFeatures(tokensAndLabelList, outLength, segIdOnesFrom);
        INDArray[] featureArray = featuresAndMaskArraysPair.getFirst();
        INDArray[] featureMaskArray = featuresAndMaskArraysPair.getSecond();


        Pair<INDArray[], INDArray[]> labelsAndMaskArraysPair = convertMiniBatchLabels(tokensAndLabelList, featureArray, outLength);
        INDArray[] labelArray = labelsAndMaskArraysPair.getFirst();
        INDArray[] labelMaskArray = labelsAndMaskArraysPair.getSecond();

        org.nd4j.linalg.dataset.MultiDataSet mds = new org.nd4j.linalg.dataset.MultiDataSet(featureArray, labelArray, featureMaskArray, labelMaskArray);
        if (preProcessor != null)
            preProcessor.preProcess(mds);

        return mds;
    }


    /**
     * For use during inference. Will convert a given list of sentences to features and feature masks as appropriate.
     *
     * @param listOnlySentences
     * @return Pair of INDArrays[], first element is feature arrays and the second is the masks array
     */
    public Pair<INDArray[], INDArray[]> featurizeSentences(List<String> listOnlySentences) {

        List<Pair<String, String>> sentencesWithNullLabel = addDummyLabel(listOnlySentences);
        SentenceListProcessed sentenceListProcessed = tokenizeMiniBatch(sentencesWithNullLabel);
        List<Pair<List<String>, String>> tokensAndLabelList = sentenceListProcessed.getTokensAndLabelList();
        int outLength = sentenceListProcessed.getMaxL();

        if (preProcessor != null) {
            Pair<INDArray[], INDArray[]> featureFeatureMasks = convertMiniBatchFeatures(tokensAndLabelList, outLength, null);
            MultiDataSet dummyMDS = new org.nd4j.linalg.dataset.MultiDataSet(featureFeatureMasks.getFirst(), null, featureFeatureMasks.getSecond(), null);
            preProcessor.preProcess(dummyMDS);
            return new Pair<>(dummyMDS.getFeatures(), dummyMDS.getFeaturesMaskArrays());
        }
        return convertMiniBatchFeatures(tokensAndLabelList, outLength, null);
    }

    /**
     * For use during inference. Will convert a given pair of a list of sentences to features and feature masks as appropriate.
     *
     * @param listOnlySentencePairs
     * @return Pair of INDArrays[], first element is feature arrays and the second is the masks array
     */
    public Pair<INDArray[], INDArray[]> featurizeSentencePairs(List<Pair<String, String>> listOnlySentencePairs) {
        Preconditions.checkState(sentencePairProvider != null, "The featurizeSentencePairs method is meant for inference with sentence pairs. Use only when the sentence pair provider is set (i.e not null).");

        List<Triple<String, String, String>> sentencePairsWithNullLabel = addDummyLabelForPairs(listOnlySentencePairs);
        SentencePairListProcessed sentencePairListProcessed = tokenizePairsMiniBatch(sentencePairsWithNullLabel);
        List<Pair<List<String>, String>> tokensAndLabelList = sentencePairListProcessed.getTokensAndLabelList();
        int outLength = sentencePairListProcessed.getMaxL();
        long[] segIdOnesFrom = sentencePairListProcessed.getSegIdOnesFrom();
        if (preProcessor != null) {
            Pair<INDArray[], INDArray[]> featuresAndMaskArraysPair = convertMiniBatchFeatures(tokensAndLabelList, outLength, segIdOnesFrom);
            MultiDataSet dummyMDS = new org.nd4j.linalg.dataset.MultiDataSet(featuresAndMaskArraysPair.getFirst(), null, featuresAndMaskArraysPair.getSecond(), null);
            preProcessor.preProcess(dummyMDS);
            return new Pair<>(dummyMDS.getFeatures(), dummyMDS.getFeaturesMaskArrays());
        }
        return convertMiniBatchFeatures(tokensAndLabelList, outLength, segIdOnesFrom);
    }

    private Pair<INDArray[], INDArray[]> convertMiniBatchFeatures(List<Pair<List<String>, String>> tokensAndLabelList, int outLength, long[] segIdOnesFrom) {
        int mbPadded = padMinibatches ? minibatchSize : tokensAndLabelList.size();
        int[][] outIdxs = new int[mbPadded][outLength];
        int[][] outMask = new int[mbPadded][outLength];
        int[][] outSegmentId = null;
        if (featureArrays == FeatureArrays.INDICES_MASK_SEGMENTID)
            outSegmentId = new int[mbPadded][outLength];
        for (int i = 0; i < tokensAndLabelList.size(); i++) {
            Pair<List<String>, String> p = tokensAndLabelList.get(i);
            List<String> t = p.getFirst();
            for (int j = 0; j < outLength && j < t.size(); j++) {
                Preconditions.checkState(vocabMap.containsKey(t.get(j)), "Unknown token encountered: token \"%s\" is not in vocabulary", t.get(j));
                int idx = vocabMap.get(t.get(j));
                outIdxs[i][j] = idx;
                outMask[i][j] = 1;
                if (segIdOnesFrom != null && j >= segIdOnesFrom[i])
                    outSegmentId[i][j] = 1;
            }
        }

        //Create actual arrays. Indices, mask, and optional segment ID
        INDArray outIdxsArr = Nd4j.createFromArray(outIdxs);
        INDArray outMaskArr = Nd4j.createFromArray(outMask);
        INDArray outSegmentIdArr;
        INDArray[] f;
        INDArray[] fm;
        if (featureArrays == FeatureArrays.INDICES_MASK_SEGMENTID) {
            outSegmentIdArr = Nd4j.createFromArray(outSegmentId);
            f = new INDArray[]{outIdxsArr, outSegmentIdArr};
            fm = new INDArray[]{outMaskArr, null};
        } else {
            f = new INDArray[]{outIdxsArr};
            fm = new INDArray[]{outMaskArr};
        }
        return new Pair<>(f, fm);
    }

    private SentenceListProcessed tokenizeMiniBatch(List<Pair<String, String>> list) {
        //Get and tokenize the sentences for this minibatch
        SentenceListProcessed sentenceListProcessed = new SentenceListProcessed(list.size());
        int longestSeq = -1;
        for (Pair<String, String> p : list) {
            List<String> tokens = tokenizeSentence(p.getFirst());
            sentenceListProcessed.addProcessedToList(new Pair<>(tokens, p.getSecond()));
            longestSeq = Math.max(longestSeq, tokens.size());
        }
        //Determine output array length...
        int outLength;
        switch (lengthHandling) {
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
        sentenceListProcessed.setMaxL(outLength);
        return sentenceListProcessed;
    }

    private SentencePairListProcessed tokenizePairsMiniBatch(List<Triple<String, String, String>> listPairs) {
        SentencePairListProcessed sentencePairListProcessed = new SentencePairListProcessed(listPairs.size());
        for (Triple<String, String, String> t : listPairs) {
            List<String> tokensL = tokenizeSentence(t.getFirst(), true);
            List<String> tokensR = tokenizeSentence(t.getSecond(), true);
            List<String> tokens = new ArrayList<>(maxTokens);
            int maxLength = maxTokens;
            if (prependToken != null)
                maxLength--;
            if (appendToken != null)
                maxLength -= 2;
            if (tokensL.size() + tokensR.size() > maxLength) {
                boolean shortOnL = tokensL.size() < tokensR.size();
                int shortSize = Math.min(tokensL.size(), tokensR.size());
                if (shortSize > maxLength / 2) {
                    //both lists need to be sliced
                    tokensL.subList(maxLength / 2, tokensL.size()).clear(); //if maxsize/2 is odd pop extra on L side to match implementation in TF
                    tokensR.subList(maxLength - maxLength / 2, tokensR.size()).clear();
                } else {
                    //slice longer list
                    if (shortOnL) {
                        //longer on R - slice R
                        tokensR.subList(maxLength - tokensL.size(), tokensR.size()).clear();
                    } else {
                        //longer on L - slice L
                        tokensL.subList(maxLength - tokensR.size(), tokensL.size()).clear();
                    }
                }
            }
            if (prependToken != null)
                tokens.add(prependToken);
            tokens.addAll(tokensL);
            if (appendToken != null)
                tokens.add(appendToken);
            int segIdOnesFrom = tokens.size();
            tokens.addAll(tokensR);
            if (appendToken != null)
                tokens.add(appendToken);
            sentencePairListProcessed.addProcessedToList(segIdOnesFrom, new Pair<>(tokens, t.getThird()));
        }
        sentencePairListProcessed.setMaxL(maxTokens);
        return sentencePairListProcessed;
    }

    private Pair<INDArray[], INDArray[]> convertMiniBatchLabels(List<Pair<List<String>, String>> tokenizedSentences, INDArray[] featureArray, int outLength) {
        INDArray[] l = new INDArray[1];
        INDArray[] lm;
        int mbSize = tokenizedSentences.size();
        int mbPadded = padMinibatches ? minibatchSize : tokenizedSentences.size();
        if (task == Task.SEQ_CLASSIFICATION) {
            //Sequence classification task: output is 2d, one-hot, shape [minibatch, numClasses]
            int numClasses;
            int[] classLabels = new int[mbPadded];
            if (sentenceProvider != null) {
                numClasses = sentenceProvider.numLabelClasses();
                List<String> labels = sentenceProvider.allLabels();
                for (int i = 0; i < mbSize; i++) {
                    String lbl = tokenizedSentences.get(i).getRight();
                    classLabels[i] = labels.indexOf(lbl);
                    Preconditions.checkState(classLabels[i] >= 0, "Provided label \"%s\" for sentence does not exist in set of classes/categories", lbl);
                }
            } else if (sentencePairProvider != null) {
                numClasses = sentencePairProvider.numLabelClasses();
                List<String> labels = sentencePairProvider.allLabels();
                for (int i = 0; i < mbSize; i++) {
                    String lbl = tokenizedSentences.get(i).getRight();
                    classLabels[i] = labels.indexOf(lbl);
                    Preconditions.checkState(classLabels[i] >= 0, "Provided label \"%s\" for sentence does not exist in set of classes/categories", lbl);
                }
            } else {
                throw new RuntimeException();
            }
            l[0] = Nd4j.create(DataType.FLOAT, mbPadded, numClasses);
            for (int i = 0; i < mbSize; i++) {
                l[0].putScalar(i, classLabels[i], 1.0);
            }
            lm = null;
            if (padMinibatches && mbSize != mbPadded) {
                INDArray a = Nd4j.zeros(DataType.FLOAT, mbPadded, 1);
                lm = new INDArray[]{a};
                a.get(NDArrayIndex.interval(0, mbSize), NDArrayIndex.all()).assign(1);
            }
        } else if (task == Task.UNSUPERVISED) {
            //Unsupervised, masked language model task
            //Output is either 2d, or 3d depending on settings
            if (vocabKeysAsList == null) {
                String[] arr = new String[vocabMap.size()];
                for (Map.Entry<String, Integer> e : vocabMap.entrySet()) {
                    arr[e.getValue()] = e.getKey();
                }
                vocabKeysAsList = Arrays.asList(arr);
            }


            int vocabSize = vocabMap.size();
            INDArray labelArr;
            INDArray lMask = Nd4j.zeros(DataType.INT, mbPadded, outLength);
            if (unsupervisedLabelFormat == UnsupervisedLabelFormat.RANK2_IDX) {
                labelArr = Nd4j.create(DataType.INT, mbPadded, outLength);
            } else if (unsupervisedLabelFormat == UnsupervisedLabelFormat.RANK3_NCL) {
                labelArr = Nd4j.create(DataType.FLOAT, mbPadded, vocabSize, outLength);
            } else if (unsupervisedLabelFormat == UnsupervisedLabelFormat.RANK3_LNC) {
                labelArr = Nd4j.create(DataType.FLOAT, outLength, mbPadded, vocabSize);
            } else {
                throw new IllegalStateException("Unknown unsupervised label format: " + unsupervisedLabelFormat);
            }

            for (int i = 0; i < mbSize; i++) {
                List<String> tokens = tokenizedSentences.get(i).getFirst();
                Pair<List<String>, boolean[]> p = masker.maskSequence(tokens, maskToken, vocabKeysAsList);
                List<String> maskedTokens = p.getFirst();
                boolean[] predictionTarget = p.getSecond();
                int seqLen = Math.min(predictionTarget.length, outLength);
                for (int j = 0; j < seqLen; j++) {
                    if (predictionTarget[j]) {
                        String oldToken = tokenizedSentences.get(i).getFirst().get(j);  //This is target
                        int targetTokenIdx = vocabMap.get(oldToken);
                        if (unsupervisedLabelFormat == UnsupervisedLabelFormat.RANK2_IDX) {
                            labelArr.putScalar(i, j, targetTokenIdx);
                        } else if (unsupervisedLabelFormat == UnsupervisedLabelFormat.RANK3_NCL) {
                            labelArr.putScalar(i, j, targetTokenIdx, 1.0);
                        } else if (unsupervisedLabelFormat == UnsupervisedLabelFormat.RANK3_LNC) {
                            labelArr.putScalar(j, i, targetTokenIdx, 1.0);
                        }

                        lMask.putScalar(i, j, 1.0);

                        //Also update previously created feature label indexes:
                        String newToken = maskedTokens.get(j);
                        int newTokenIdx = vocabMap.get(newToken);
                        //first element of features is outIdxsArr
                        featureArray[0].putScalar(i, j, newTokenIdx);
                    }
                }
            }
            l[0] = labelArr;
            lm = new INDArray[1];
            lm[0] = lMask;
        } else {
            throw new IllegalStateException("Task not yet implemented: " + task);
        }
        return new Pair<>(l, lm);
    }

    private List<String> tokenizeSentence(String sentence) {
        return tokenizeSentence(sentence, false);
    }

    private List<String> tokenizeSentence(String sentence, boolean ignorePrependAppend) {
        Tokenizer t = tokenizerFactory.create(sentence);

        List<String> tokens = new ArrayList<>();
        if (prependToken != null && !ignorePrependAppend)
            tokens.add(prependToken);

        while (t.hasMoreTokens()) {
            String token = t.nextToken();
            tokens.add(token);
        }
        if (appendToken != null && !ignorePrependAppend)
            tokens.add(appendToken);
        return tokens;
    }


    private List<Pair<String, String>> addDummyLabel(List<String> listOnlySentences) {
        List<Pair<String, String>> list = new ArrayList<>(listOnlySentences.size());
        for (String s : listOnlySentences) {
            list.add(new Pair<String, String>(s, null));
        }
        return list;
    }

    private List<Triple<String, String, String>> addDummyLabelForPairs(List<Pair<String, String>> listOnlySentencePairs) {
        List<Triple<String, String, String>> list = new ArrayList<>(listOnlySentencePairs.size());
        for (Pair<String, String> p : listOnlySentencePairs) {
            list.add(new Triple<String, String, String>(p.getFirst(), p.getSecond(), null));
        }
        return list;
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
        if (sentenceProvider != null) {
            sentenceProvider.reset();
        }
    }

    public static Builder builder() {
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
        protected LabeledPairSentenceProvider sentencePairProvider = null;
        protected FeatureArrays featureArrays = FeatureArrays.INDICES_MASK_SEGMENTID;
        protected Map<String, Integer> vocabMap;   //TODO maybe use Eclipse ObjectIntHashMap for fewer objects?
        protected BertSequenceMasker masker = new BertMaskedLMMasker();
        protected UnsupervisedLabelFormat unsupervisedLabelFormat;
        protected String maskToken;
        protected String prependToken;
        protected String appendToken;

        /**
         * Specify the {@link Task} the iterator should be set up for. See {@link BertIterator} for more details.
         */
        public Builder task(Task task) {
            this.task = task;
            return this;
        }

        /**
         * Specify the TokenizerFactory to use.
         * For BERT, typically {@link org.deeplearning4j.text.tokenization.tokenizerfactory.BertWordPieceTokenizerFactory}
         * is used
         */
        public Builder tokenizer(TokenizerFactory tokenizerFactory) {
            this.tokenizerFactory = tokenizerFactory;
            return this;
        }

        /**
         * Specifies how the sequence length of the output data should be handled. See {@link BertIterator} for more details.
         *
         * @param lengthHandling Length handling
         * @param maxLength      Not used if LengthHandling is set to {@link LengthHandling#ANY_LENGTH}
         * @return
         */
        public Builder lengthHandling(@NonNull LengthHandling lengthHandling, int maxLength) {
            this.lengthHandling = lengthHandling;
            this.maxTokens = maxLength;
            return this;
        }

        /**
         * Minibatch size to use (number of examples to train on for each iteration)
         * See also: {@link #padMinibatches}
         *
         * @param minibatchSize Minibatch size
         */
        public Builder minibatchSize(int minibatchSize) {
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
        public Builder padMinibatches(boolean padMinibatches) {
            this.padMinibatches = padMinibatches;
            return this;
        }

        /**
         * Set the preprocessor to be used on the MultiDataSets before returning them. Default: none (null)
         */
        public Builder preProcessor(MultiDataSetPreProcessor preProcessor) {
            this.preProcessor = preProcessor;
            return this;
        }

        /**
         * Specify the source of the data for classification.
         */
        public Builder sentenceProvider(LabeledSentenceProvider sentenceProvider) {
            this.sentenceProvider = sentenceProvider;
            return this;
        }

        /**
         * Specify the source of the data for classification on sentence pairs.
         */
        public Builder sentencePairProvider(LabeledPairSentenceProvider sentencePairProvider) {
            this.sentencePairProvider = sentencePairProvider;
            return this;
        }

        /**
         * Specify what arrays should be returned. See {@link BertIterator} for more details.
         */
        public Builder featureArrays(FeatureArrays featureArrays) {
            this.featureArrays = featureArrays;
            return this;
        }

        /**
         * Provide the vocabulary as a map. Keys are the words in the vocabulary, and values are the indices of those
         * words. For indices, they should be in range 0 to vocabMap.size()-1 inclusive.<br>
         * If using {@link BertWordPieceTokenizerFactory},
         * this can be obtained using {@link BertWordPieceTokenizerFactory#getVocab()}
         */
        public Builder vocabMap(Map<String, Integer> vocabMap) {
            this.vocabMap = vocabMap;
            return this;
        }

        /**
         * Used only for unsupervised training (i.e., when task is set to {@link Task#UNSUPERVISED} for learning a
         * masked language model. This can be used to customize how the masking is performed.<br>
         * Default: {@link BertMaskedLMMasker}
         */
        public Builder masker(BertSequenceMasker masker) {
            this.masker = masker;
            return this;
        }

        /**
         * Used only for unsupervised training (i.e., when task is set to {@link Task#UNSUPERVISED} for learning a
         * masked language model. Used to specify the format that the labels should be returned in.
         * See {@link BertIterator} for more details.
         */
        public Builder unsupervisedLabelFormat(UnsupervisedLabelFormat labelFormat) {
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
        public Builder maskToken(String maskToken) {
            this.maskToken = maskToken;
            return this;
        }

        /**
         * Prepend the specified token to the sequences, when doing supervised training.<br>
         * i.e., any token sequences will have this added at the start.<br>
         * Some BERT/Transformer models may need this - for example sequences starting with a "[CLS]" token.<br>
         * No token is prepended by default.
         *
         * @param prependToken The token to start each sequence with (null: no token will be prepended)
         */
        public Builder prependToken(String prependToken) {
            this.prependToken = prependToken;
            return this;
        }

        /**
         * Append the specified token to the sequences, when doing training on sentence pairs.<br>
         * Generally "[SEP]" is used
         * No token in appended by default.
         *
         * @param appendToken Token at end of each sentence for pairs of sentences (null: no token will be appended)
         * @return
         */
        public Builder appendToken(String appendToken) {
            this.appendToken = appendToken;
            return this;
        }

        public BertIterator build() {
            Preconditions.checkState(task != null, "No task has been set. Use .task(BertIterator.Task.X) to set the task to be performed");
            Preconditions.checkState(tokenizerFactory != null, "No tokenizer factory has been set. A tokenizer factory (such as BertWordPieceTokenizerFactory) is required");
            Preconditions.checkState(vocabMap != null, "Cannot create iterator: No vocabMap has been set. Use Builder.vocabMap(Map<String,Integer>) to set");
            Preconditions.checkState(task != Task.UNSUPERVISED || masker != null, "If task is UNSUPERVISED training, a masker must be set via masker(BertSequenceMasker) method");
            Preconditions.checkState(task != Task.UNSUPERVISED || unsupervisedLabelFormat != null, "If task is UNSUPERVISED training, a label format must be set via masker(BertSequenceMasker) method");
            Preconditions.checkState(task != Task.UNSUPERVISED || maskToken != null, "If task is UNSUPERVISED training, the mask token in the vocab (such as \"[MASK]\" must be specified");
            if (sentencePairProvider != null) {
                Preconditions.checkState(task == Task.SEQ_CLASSIFICATION, "Currently only supervised sequence classification is set up with sentence pairs. \".task(BertIterator.Task.SEQ_CLASSIFICATION)\" is required with a sentence pair provider");
                Preconditions.checkState(featureArrays == FeatureArrays.INDICES_MASK_SEGMENTID, "Currently only supervised sequence classification is set up with sentence pairs. \".featureArrays(FeatureArrays.INDICES_MASK_SEGMENTID)\" is required with a sentence pair provider");
                Preconditions.checkState(lengthHandling == LengthHandling.FIXED_LENGTH, "Currently only fixed length is supported for sentence pairs. \".lengthHandling(BertIterator.LengthHandling.FIXED_LENGTH, maxLength)\" is required with a sentence pair provider");
                Preconditions.checkState(sentencePairProvider != null, "Provide either a sentence provider or a sentence pair provider. Both cannot be non null");
            }
            if (appendToken != null) {
                Preconditions.checkState(sentencePairProvider != null, "Tokens are only appended with sentence pairs. Sentence pair provider is not set. Set sentence pair provider.");
            }
            return new BertIterator(this);
        }
    }

    private static class SentencePairListProcessed {
        private int listLength = 0;

        @Getter
        private long[] segIdOnesFrom;
        private int cursor = 0;
        private SentenceListProcessed sentenceListProcessed;

        private SentencePairListProcessed(int listLength) {
            this.listLength = listLength;
            segIdOnesFrom = new long[listLength];
            sentenceListProcessed = new SentenceListProcessed(listLength);
        }

        private void addProcessedToList(long segIdIdx, Pair<List<String>, String> tokenizedSentencePairAndLabel) {
            segIdOnesFrom[cursor] = segIdIdx;
            sentenceListProcessed.addProcessedToList(tokenizedSentencePairAndLabel);
            cursor++;
        }

        private void setMaxL(int maxL) {
            sentenceListProcessed.setMaxL(maxL);
        }

        private int getMaxL() {
            return sentenceListProcessed.getMaxL();
        }

        private List<Pair<List<String>, String>> getTokensAndLabelList() {
            return sentenceListProcessed.getTokensAndLabelList();
        }
    }

    private static class SentenceListProcessed {
        private int listLength;

        @Getter
        @Setter
        private int maxL;

        @Getter
        private List<Pair<List<String>, String>> tokensAndLabelList;

        private SentenceListProcessed(int listLength) {
            this.listLength = listLength;
            tokensAndLabelList = new ArrayList<>(listLength);
        }

        private void addProcessedToList(Pair<List<String>, String> tokenizedSentenceAndLabel) {
            tokensAndLabelList.add(tokenizedSentenceAndLabel);
        }
    }
}
