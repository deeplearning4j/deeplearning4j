package org.nd4j.linalg.dataset.api;

import com.google.common.base.Function;
import org.apache.commons.math3.random.RandomGenerator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.indexing.Condition;

import java.io.Serializable;
import java.util.Iterator;
import java.util.List;

/**
 * Created by agibsonccc on 8/26/14.
 */
public interface DataSet extends Iterable<org.nd4j.linalg.dataset.DataSet>,Serializable {
    INDArray getFeatures();


    void apply(Condition condition,Function<Number,Number> function);

    void setFeatures(INDArray features);

    void setLabels(INDArray labels);

    org.nd4j.linalg.dataset.DataSet copy();

    org.nd4j.linalg.dataset.DataSet reshape(int rows, int cols);

    void multiplyBy(double num);

    void divideBy(int num);

    void shuffle();

    void squishToRange(double min, double max);

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

    List<List<org.nd4j.linalg.dataset.DataSet>> batchBy(int num);

    org.nd4j.linalg.dataset.DataSet filterBy(int[] labels);

    void filterAndStrip(int[] labels);

    List<org.nd4j.linalg.dataset.DataSet> dataSetBatches(int num);

    List<List<org.nd4j.linalg.dataset.DataSet>> sortAndBatchByNumLabels();

    List<List<org.nd4j.linalg.dataset.DataSet>> batchByNumLabels();

    List<org.nd4j.linalg.dataset.DataSet> asList();

    SplitTestAndTrain splitTestAndTrain(int numHoldout);

    INDArray getLabels();

    INDArray getFeatureMatrix();

    void sortByLabel();

    void addRow(org.nd4j.linalg.dataset.DataSet d, int i);

    INDArray exampleSums();

    INDArray exampleMaxs();

    INDArray exampleMeans();

    org.nd4j.linalg.dataset.DataSet sample(int numSamples);

    org.nd4j.linalg.dataset.DataSet sample(int numSamples, RandomGenerator rng);

    org.nd4j.linalg.dataset.DataSet sample(int numSamples, boolean withReplacement);

    org.nd4j.linalg.dataset.DataSet sample(int numSamples, RandomGenerator rng, boolean withReplacement);

    void roundToTheNearest(int roundTo);

    int numOutcomes();

    int numExamples();

    List<String> getLabelNames();

    void setLabelNames(List<String> labelNames);

    List<String> getColumnNames();

    void setColumnNames(List<String> columnNames);

    @Override
    Iterator<org.nd4j.linalg.dataset.DataSet> iterator();
}
