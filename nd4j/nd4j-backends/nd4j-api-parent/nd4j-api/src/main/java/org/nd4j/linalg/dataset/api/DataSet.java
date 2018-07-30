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

package org.nd4j.linalg.dataset.api;

import com.google.common.base.Function;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.indexing.conditions.Condition;

import java.io.File;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.Serializable;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

/**
 * Created by agibsonccc on 8/26/14.
 */
public interface DataSet extends Iterable<org.nd4j.linalg.dataset.DataSet>, Serializable {


    DataSet getRange(int from, int to);

    /**
     * Load the contents of the DataSet from the specified InputStream. The current contents of the DataSet (if any) will be replaced.<br>
     * The InputStream should contain a DataSet that has been serialized with {@link #save(OutputStream)}
     *
     * @param from InputStream to load the DataSet from
     */
    void load(InputStream from);

    /**
     * Load the contents of the DataSet from the specified File. The current contents of the DataSet (if any) will be replaced.<br>
     * The InputStream should contain a DataSet that has been serialized with {@link #save(File)}
     *
     * @param from File to load the DataSet from
     */
    void load(File from);

    /**
     * Write the contents of this DataSet to the specified OutputStream
     *
     * @param to OutputStream to save the DataSet to
     */
    void save(OutputStream to);

    /**
     * Save this DataSet to a file. Can be loaded again using {@link }
     *
     * @param to    File to sa
     */
    void save(File to);

    @Deprecated
    DataSetIterator iterateWithMiniBatches();

    String id();

    /**
     * Returns the features array for the DataSet
     *
     * @return features array
     */
    INDArray getFeatures();

    /**
     * Set the features array for the DataSet
     *
     * @param features    Features to set
     */
    void setFeatures(INDArray features);

    /**
     * Calculate and return a count of each label, by index.
     * Assumes labels are a one-hot INDArray, for classification
     *
     * @return Map of countsn
     */
    Map<Integer, Double> labelCounts();

    void apply(Condition condition, Function<Number, Number> function);

    /**
     * Create a copy of the DataSet
     *
     * @return Copy of the DataSet
     */
    org.nd4j.linalg.dataset.DataSet copy();

    org.nd4j.linalg.dataset.DataSet reshape(int rows, int cols);

    /**
     * Multiply the features by a scalar
     */
    void multiplyBy(double num);

    /**
     * Divide the features by a scalar
     */
    void divideBy(int num);

    /**
     * Shuffle the order of the rows in the DataSet. Note that this generally won't make any difference in practice
     * unless the DataSet is later split.
     */
    void shuffle();

    void squishToRange(double min, double max);

    void scaleMinAndMax(double min, double max);

    void scale();

    void addFeatureVector(INDArray toAdd);

    void addFeatureVector(INDArray feature, int example);

    /**
     * Normalize this DataSet to mean 0, stdev 1 per input.
     * This calculates statistics based on the values in a single DataSet only.
     * For normalization over multiple DataSet objects, use {@link org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize}
     */
    void normalize();

    void binarize();

    void binarize(double cutoff);

    /**
     * @deprecated Use {@link org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize}
     */
    @Deprecated
    void normalizeZeroMeanZeroUnitVariance();

    /**
     * Number of input values - i.e., size of the features INDArray per example
     */
    int numInputs();

    void validate();

    int outcome();

    void setNewNumberOfLabels(int labels);

    void setOutcome(int example, int label);

    org.nd4j.linalg.dataset.DataSet get(int i);

    org.nd4j.linalg.dataset.DataSet get(int[] i);

    List<org.nd4j.linalg.dataset.DataSet> batchBy(int num);

    org.nd4j.linalg.dataset.DataSet filterBy(int[] labels);

    void filterAndStrip(int[] labels);

    /**
     * @deprecated prefer {@link #batchBy(int)}
     */
    @Deprecated
    List<org.nd4j.linalg.dataset.DataSet> dataSetBatches(int num);

    List<org.nd4j.linalg.dataset.DataSet> sortAndBatchByNumLabels();

    List<org.nd4j.linalg.dataset.DataSet> batchByNumLabels();

    /**
     * Extract each example in the DataSet into its own DataSet object, and return all of them as a list
     * @return List of DataSet objects, each with 1 example only
     */
    List<org.nd4j.linalg.dataset.DataSet> asList();

    SplitTestAndTrain splitTestAndTrain(int numHoldout, java.util.Random rnd);

    SplitTestAndTrain splitTestAndTrain(int numHoldout);

    INDArray getLabels();

    void setLabels(INDArray labels);

    /**
     * Equivalent to {@link #getFeatures()}
     * @deprecated Use {@link #getFeatures()}
     */
    @Deprecated
    INDArray getFeatureMatrix();

    void sortByLabel();

    void addRow(org.nd4j.linalg.dataset.DataSet d, int i);

    INDArray exampleSums();

    INDArray exampleMaxs();

    INDArray exampleMeans();

    org.nd4j.linalg.dataset.DataSet sample(int numSamples);

    org.nd4j.linalg.dataset.DataSet sample(int numSamples, Random rng);

    org.nd4j.linalg.dataset.DataSet sample(int numSamples, boolean withReplacement);

    org.nd4j.linalg.dataset.DataSet sample(int numSamples, Random rng, boolean withReplacement);

    void roundToTheNearest(int roundTo);

    /**
     * Returns the number of outcomes (size of the labels array for each example)
     */
    int numOutcomes();

    /**
     * Number of examples in the DataSet
     */
    int numExamples();

    @Deprecated
    List<String> getLabelNames();

    List<String> getLabelNamesList();

    String getLabelName(int idx);

    List<String> getLabelNames(INDArray idxs);

    void setLabelNames(List<String> labelNames);

    List<String> getColumnNames();

    void setColumnNames(List<String> columnNames);

    /**
     * SplitV the DataSet into two DataSets randomly
     * @param fractionTrain    Fraction (in range 0 to 1) of examples to be returned in the training DataSet object
     */
    SplitTestAndTrain splitTestAndTrain(double fractionTrain);

    @Override
    Iterator<org.nd4j.linalg.dataset.DataSet> iterator();

    /**
     * Input mask array: a mask array for input, where each value is in {0,1} in order to specify whether an input is
     * actually present or not. Typically used for situations such as RNNs with variable length inputs
     *
     * @return Input mask array
     */
    INDArray getFeaturesMaskArray();

    /**
     * Set the features mask array in this DataSet
     */
    void setFeaturesMaskArray(INDArray inputMask);

    /**
     * Labels (output) mask array: a mask array for input, where each value is in {0,1} in order to specify whether an
     * output is actually present or not. Typically used for situations such as RNNs with variable length inputs or many-
     * to-one situations.
     *
     * @return Labels (output) mask array
     */
    INDArray getLabelsMaskArray();

    /**
     * Set the labels mask array in this data set
     */
    void setLabelsMaskArray(INDArray labelsMask);

    /**
     * Whether the labels or input (features) mask arrays are present for this DataSet
     */
    boolean hasMaskArrays();

    /**
     * Set the metadata for this DataSet<br>
     * By convention: the metadata can be any serializable object, one per example in the DataSet
     *
     * @param exampleMetaData Example metadata to set
     */
    void setExampleMetaData(List<? extends Serializable> exampleMetaData);

    /**
     * Get the example metadata, or null if no metadata has been set<br>
     * Note: this method results in an unchecked cast - care should be taken when using this!
     *
     * @param metaDataType Class of the metadata (used for opType information)
     * @param <T>          Type of metadata
     * @return List of metadata objects
     */
    <T extends Serializable> List<T> getExampleMetaData(Class<T> metaDataType);

    /**
     * Get the example metadata, or null if no metadata has been set
     *
     * @return List of metadata instances
     * @see {@link #getExampleMetaData(Class)} for convenience method for types
     */
    List<Serializable> getExampleMetaData();

    /**
     * This method returns memory used by this DataSet
     * @return
     */
    long getMemoryFootprint();

    /**
     * This method migrates this DataSet into current Workspace (if any)
     */
    void migrate();

    /**
     * This method detaches this DataSet from current Workspace (if any)
     */
    void detach();

    /**
     * @return true if the DataSet object is empty (no features, labels, or masks)
     */
    boolean isEmpty();
}
