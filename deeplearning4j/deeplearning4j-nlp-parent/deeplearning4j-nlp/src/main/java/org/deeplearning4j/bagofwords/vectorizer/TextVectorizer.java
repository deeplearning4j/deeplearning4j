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

package org.deeplearning4j.bagofwords.vectorizer;

import org.deeplearning4j.datasets.vectorizer.Vectorizer;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.text.invertedindex.InvertedIndex;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

import java.io.File;
import java.io.InputStream;
import java.util.List;

/**
 * Vectorizes text
 * @author Adam Gibson
 */
public interface TextVectorizer extends Vectorizer {


    /**
     * Sampling for building mini batches
     * @return the sampling
     */
    //double sample();

    /**
     * For word vectors, this is the batch size for how to partition documents
     * in to workloads
     * @return the batchsize for partitioning documents in to workloads
     */
    //int batchSize();

    /**
     * The vocab sorted in descending order
     * @return the vocab sorted in descending order
     */
    VocabCache<VocabWord> getVocabCache();


    /**
     * Text coming from an input stream considered as one document
     * @param is the input stream to read from
     * @param label the label to assign
     * @return a dataset with a applyTransformToDestination of weights(relative to impl; could be word counts or tfidf scores)
     */
    DataSet vectorize(InputStream is, String label);

    /**
     * Vectorizes the passed in text treating it as one document
     * @param text the text to vectorize
     * @param label the label of the text
     * @return a dataset with a transform of weights(relative to impl; could be word counts or tfidf scores)
     */
    DataSet vectorize(String text, String label);

    /**
     * Train the model
     */
    void fit();

    /**
     *
     * @param input the text to vectorize
     * @param label the label of the text
     * @return {@link DataSet} with a applyTransformToDestination of
     *          weights(relative to impl; could be word counts or tfidf scores)
     */
    DataSet vectorize(File input, String label);


    /**
     * Transforms the matrix
     * @param text text to transform
     * @return {@link INDArray}
     */
    INDArray transform(String text);

    /**
     * Transforms the matrix
     * @param tokens
     * @return
     */
    INDArray transform(List<String> tokens);

    /**
     * Returns the number of words encountered so far
     * @return the number of words encountered so far
     */
    long numWordsEncountered();

    /**
     * Inverted index
     * @return the inverted index for this vectorizer
     */
    InvertedIndex<VocabWord> getIndex();
}
