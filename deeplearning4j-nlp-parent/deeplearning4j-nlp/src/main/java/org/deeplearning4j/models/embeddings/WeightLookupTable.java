/*-
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.models.embeddings;

import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.plot.BarnesHutTsne;
import org.deeplearning4j.ui.UiConnectionInfo;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;
import java.io.Serializable;
import java.util.Iterator;
import java.util.concurrent.atomic.AtomicLong;

/**
 * General weight lookup table
 *
 * @author Adam Gibson
 */
public interface WeightLookupTable<T extends SequenceElement> extends Serializable {

    /**
     * Returns unique ID of this table.
     * Used for JointStorage/DistributedLookupTable mechanics
     *
     * @return ID of this table
     */
    Long getTableId();

    /**
     * Set's table Id.
     * Please note, it should be unique withing Joint/Distributed LookupTable
     *
     * @param tableId
     */
    void setTableId(Long tableId);

    /**
     * The layer size for the lookup table
     * @return the layer size for the lookup table
     */
    int layerSize();

    /**
     * Returns gradient for specified word
     * @param column
     * @param gradient
     * @return
     */
    double getGradient(int column, double gradient);

    /**
     * Clear out all weights regardless
     * @param reset
     */
    void resetWeights(boolean reset);

    /**
     * Render the words via TSNE
     * @param tsne the tsne to use
     */
    void plotVocab(BarnesHutTsne tsne, int numWords, UiConnectionInfo connectionInfo);

    /**
     * Render the words via TSNE
     * @param tsne the tsne to use
     */
    void plotVocab(BarnesHutTsne tsne, int numWords, File file);

    /**
     * Render the words via tsne
     */
    void plotVocab(int numWords, UiConnectionInfo connectionInfo);

    /**
     * Render the words via tsne
     */
    void plotVocab(int numWords, File file);

    /**
     *
     * @param codeIndex
     * @param code
     */
    void putCode(int codeIndex, INDArray code);

    /**
     * Loads the co-occurrences for the given codes
     * @param codes the codes to load
     * @return an ndarray of code.length by layerSize
     */
    INDArray loadCodes(int[] codes);

    /**
     * Iterate on the given 2 vocab words
     * @param w1 the first word to iterate on
     * @param w2 the second word to iterate on
     */
    @Deprecated
    void iterate(T w1, T w2);

    /**
     * Iterate on the given 2 vocab words
     * @param w1 the first word to iterate on
     * @param w2 the second word to iterate on
     * @param nextRandom nextRandom for sampling
     * @param alpha the alpha to use for learning
     */
    @Deprecated
    void iterateSample(T w1, T w2, AtomicLong nextRandom, double alpha);


    /**
     * Inserts a word vector
     * @param word the word to insert
     * @param vector the vector to insert
     */
    void putVector(String word, INDArray vector);

    /**
     *
     * @param word
     * @return
     */
    INDArray vector(String word);

    /**
     * Reset the weights of the cache
     */
    void resetWeights();


    /**
     * Sets the learning rate
     * @param lr
     */
    void setLearningRate(double lr);

    /**
     * Iterates through all of the vectors in the cache
     * @return an iterator for all vectors in the cache
     */
    Iterator<INDArray> vectors();

    INDArray getWeights();

    /**
     * Returns corresponding vocabulary
     */
    VocabCache<T> getVocabCache();
}
