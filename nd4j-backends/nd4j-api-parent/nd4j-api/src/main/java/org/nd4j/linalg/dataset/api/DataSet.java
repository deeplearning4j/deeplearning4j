/*
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
 *
 */

package org.nd4j.linalg.dataset.api;

import com.google.common.base.Function;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.indexing.conditions.Condition;

import java.io.File;
import java.io.Serializable;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

/**
 * Created by agibsonccc on 8/26/14.
 */
public interface DataSet extends Iterable<org.nd4j.linalg.dataset.DataSet>, Serializable {



    DataSet getRange(int from,int to);

    void load(File from);

    void save(File to);

    DataSetIterator iterateWithMiniBatches();

    String id();

    INDArray getFeatures();

    void setFeatures(INDArray features);

    Map<Integer, Double> labelCounts();

    void apply(Condition condition, Function<Number, Number> function);

    org.nd4j.linalg.dataset.DataSet copy();

    org.nd4j.linalg.dataset.DataSet reshape(int rows, int cols);

    void multiplyBy(double num);

    void divideBy(int num);

    void shuffle();

    void squishToRange(double min, double max);

    void scaleMinAndMax(double min, double max);

    void scale();

    void addFeatureVector(INDArray toAdd);

    void addFeatureVector(INDArray feature, int example);

    void normalize();

    void binarize();

    void binarize(double cutoff);

    void normalizeZeroMeanZeroUnitVariance();

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

    List<org.nd4j.linalg.dataset.DataSet> asList();

    SplitTestAndTrain splitTestAndTrain(int numHoldout, java.util.Random rnd);

    SplitTestAndTrain splitTestAndTrain(int numHoldout);

    INDArray getLabels();

    void setLabels(INDArray labels);

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

    int numOutcomes();

    int numExamples();

    List<String> getLabelNames();

    void setLabelNames(List<String> labelNames);

    List<String> getColumnNames();

    void setColumnNames(List<String> columnNames);


    SplitTestAndTrain splitTestAndTrain(double percentTrain);

    @Override
    Iterator<org.nd4j.linalg.dataset.DataSet> iterator();

    /** Input mask array: a mask array for input, where each value is in {0,1} in order to specify whether an input is
     *  actually present or not. Typically used for situations such as RNNs with variable length inputs
      * @return Input mask array
     */
    INDArray getFeaturesMaskArray();

    void setFeaturesMaskArray(INDArray inputMask);

    /** Labels (output) mask array: a mask array for input, where each value is in {0,1} in order to specify whether an
     * output is actually present or not. Typically used for situations such as RNNs with variable length inputs or many-
     * to-one situations.
     * @return Labels (output) mask array
     */
    INDArray getLabelsMaskArray();

    void setLabelsMaskArray(INDArray labelsMask);

    /** Whether the labels or input (features) mask arrays are present for this DataSet */
    boolean hasMaskArrays();
}
