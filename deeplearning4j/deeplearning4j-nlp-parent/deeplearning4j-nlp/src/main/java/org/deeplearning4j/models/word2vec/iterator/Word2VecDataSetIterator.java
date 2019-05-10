/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
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

package org.deeplearning4j.models.word2vec.iterator;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.inputsanitation.InputHomogenization;
import org.deeplearning4j.text.movingwindow.Window;
import org.deeplearning4j.text.movingwindow.WindowConverter;
import org.deeplearning4j.text.movingwindow.Windows;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;
import org.deeplearning4j.text.sentenceiterator.labelaware.LabelAwareSentenceIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.FeatureUtil;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * Iterates over a sentence with moving window to produce a data applyTransformToDestination
 * for word windows based on a pretrained word2vec.
 *
 * @author Adam Gibson
 */
@Slf4j
public class Word2VecDataSetIterator implements DataSetIterator {
    private Word2Vec vec;
    private LabelAwareSentenceIterator iter;
    private List<Window> cachedWindow;
    private List<String> labels;
    private int batch = 10;
    @Getter
    private DataSetPreProcessor preProcessor;

    /**
     * Allows for customization of all of the params of the iterator
     * @param vec the word2vec model to use
     * @param iter the sentence iterator to use
     * @param labels the possible labels
     * @param batch the batch size
     * @param homogenization whether to homogenize the sentences or not
     * @param addLabels whether to add labels for windows
     */
    public Word2VecDataSetIterator(Word2Vec vec, LabelAwareSentenceIterator iter, List<String> labels, int batch,
                    boolean homogenization, boolean addLabels) {
        this.vec = vec;
        this.iter = iter;
        this.labels = labels;
        this.batch = batch;
        cachedWindow = new CopyOnWriteArrayList<>();

        if (addLabels && homogenization)
            iter.setPreProcessor(new SentencePreProcessor() {
                @Override
                public String preProcess(String sentence) {
                    String label = Word2VecDataSetIterator.this.iter.currentLabel();
                    String ret = "<" + label + "> " + new InputHomogenization(sentence).transform() + " </" + label
                                    + ">";
                    return ret;
                }
            });

        else if (addLabels)
            iter.setPreProcessor(new SentencePreProcessor() {
                @Override
                public String preProcess(String sentence) {
                    String label = Word2VecDataSetIterator.this.iter.currentLabel();
                    String ret = "<" + label + ">" + sentence + "</" + label + ">";
                    return ret;
                }
            });

        else if (homogenization)
            iter.setPreProcessor(new SentencePreProcessor() {
                @Override
                public String preProcess(String sentence) {
                    String ret = new InputHomogenization(sentence).transform();
                    return ret;
                }
            });

    }

    /**
     * Initializes this iterator with homogenization and adding labels
     * and a batch size of 10
     * @param vec the vector model to use
     * @param iter the sentence iterator to use
     * @param labels the possible labels
     */
    public Word2VecDataSetIterator(Word2Vec vec, LabelAwareSentenceIterator iter, List<String> labels) {
        this(vec, iter, labels, 10);
    }

    /**
     * Initializes this iterator with homogenization and adding labels
     * @param vec the vector model to use
     * @param iter the sentence iterator to use
     * @param labels the possible labels
     * @param batch the batch size
     */
    public Word2VecDataSetIterator(Word2Vec vec, LabelAwareSentenceIterator iter, List<String> labels, int batch) {
        this(vec, iter, labels, batch, true, true);


    }

    /**
     * Like the standard next method but allows a
     * customizable number of examples returned
     *
     * @param num the number of examples
     * @return the next data applyTransformToDestination
     */
    @Override
    public DataSet next(int num) {
        if (num <= cachedWindow.size())
            return fromCached(num);
        //no more sentences, return the left over
        else if (num >= cachedWindow.size() && !iter.hasNext())
            return fromCached(cachedWindow.size());

        //need the next sentence
        else {
            while (cachedWindow.size() < num && iter.hasNext()) {
                String sentence = iter.nextSentence();
                if (sentence.isEmpty())
                    continue;
                List<Window> windows = Windows.windows(sentence, vec.getTokenizerFactory(), vec.getWindow(), vec);
                if (windows.isEmpty() && !sentence.isEmpty())
                    throw new IllegalStateException("Empty window on sentence");
                for (Window w : windows)
                    w.setLabel(iter.currentLabel());
                cachedWindow.addAll(windows);
            }

            return fromCached(num);
        }

    }

    private DataSet fromCached(int num) {
        if (cachedWindow.isEmpty()) {
            while (cachedWindow.size() < num && iter.hasNext()) {
                String sentence = iter.nextSentence();
                if (sentence.isEmpty())
                    continue;
                List<Window> windows = Windows.windows(sentence, vec.getTokenizerFactory(), vec.getWindow(), vec);
                for (Window w : windows)
                    w.setLabel(iter.currentLabel());
                cachedWindow.addAll(windows);
            }
        }


        List<Window> windows = new ArrayList<>(num);

        for (int i = 0; i < num; i++) {
            if (cachedWindow.isEmpty())
                break;
            windows.add(cachedWindow.remove(0));
        }

        if (windows.isEmpty())
            return null;



        INDArray inputs = Nd4j.create(num, inputColumns());
        for (int i = 0; i < inputs.rows(); i++) {
            inputs.putRow(i, WindowConverter.asExampleMatrix(windows.get(i), vec));
        }

        INDArray labelOutput = Nd4j.create(num, labels.size());
        for (int i = 0; i < labelOutput.rows(); i++) {
            String label = windows.get(i).getLabel();
            labelOutput.putRow(i, FeatureUtil.toOutcomeVector(labels.indexOf(label), labels.size()));
        }

        DataSet ret = new DataSet(inputs, labelOutput);
        if (preProcessor != null)
            preProcessor.preProcess(ret);

        return ret;
    }

    @Override
    public int inputColumns() {
        return vec.lookupTable().layerSize() * vec.getWindow();
    }

    @Override
    public int totalOutcomes() {
        return labels.size();
    }

    @Override
    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return false;
    }

    @Override
    public void reset() {
        iter.reset();
        cachedWindow.clear();
    }

    @Override
    public int batch() {
        return batch;
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        this.preProcessor = preProcessor;
    }

    @Override
    public List<String> getLabels() {
        return null;
    }


    /**
     * Returns {@code true} if the iteration has more elements.
     * (In other words, returns {@code true} if {@link #next} would
     * return an element rather than throwing an exception.)
     *
     * @return {@code true} if the iteration has more elements
     */
    @Override
    public boolean hasNext() {
        return iter.hasNext() || !cachedWindow.isEmpty();
    }

    /**
     * Returns the next element in the iteration.
     *
     * @return the next element in the iteration
     */
    @Override
    public DataSet next() {
        return next(batch);
    }

    /**
     * Removes from the underlying collection the last element returned
     * by this iterator (optional operation).  This method can be called
     * only once per call to {@link #next}.  The behavior of an iterator
     * is unspecified if the underlying collection is modified while the
     * iteration is in progress in any way other than by calling this
     * method.
     *
     * @throws UnsupportedOperationException if the {@code remove}
     *                                       operation is not supported by this iterator
     * @throws IllegalStateException         if the {@code next} method has not
     *                                       yet been called, or the {@code remove} method has already
     *                                       been called after the last call to the {@code next}
     *                                       method
     */
    @Override
    public void remove() {
        throw new UnsupportedOperationException();
    }
}
