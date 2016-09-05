package org.deeplearning4j.datasets.iterator;

import lombok.NonNull;
import org.deeplearning4j.berkeley.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;
import java.util.concurrent.LinkedBlockingQueue;

/**
 *  This is simple DataSetIterator implementation, that builds DataSetIterator out of INDArray/float[]/double[] pairs.
 *  Suitable for model feeding with externally originated data.
 *
 *  PLEASE NOTE: If total number of input elements % batchSize != 0, reminder will be ignored
 *
 * @author raver119@gmail.com
 */
public abstract class AbstractDataSetIterator<T> implements DataSetIterator {
    private DataSetPreProcessor preProcessor;
    private transient Iterable<Pair<T, T>> iterable;
    private transient Iterator<Pair<T, T>> iterator;

    private final int batchSize;
    private final Queue<DataSet> queue = new LinkedBlockingQueue<>(1);
    private List<String> labels;
    private int numFeatures = -1;
    private int numLabels = -1;

    protected AbstractDataSetIterator(@NonNull Iterable<Pair<T, T>> iterable, int batchSize) {
        if (batchSize < 1)
            throw new IllegalStateException("batchSize can't be < 1");

        this.iterable = iterable;
        this.iterator = this.iterable.iterator();
        this.batchSize = batchSize;

        fillQueue();
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
        throw new IllegalStateException("next(int) isn't supported for this DataSetIterator");
    }

    /**
     * Total examples in the iterator
     *
     * @return
     */
    @Override
    public int totalExamples() {
        return 0;
    }

    /**
     * Input columns for the dataset
     *
     * @return
     */
    @Override
    public int inputColumns() {
        return numFeatures;
    }

    /**
     * The number of labels for the dataset
     *
     * @return
     */
    @Override
    public int totalOutcomes() {
        return numLabels;
    }

    @Override
    public boolean resetSupported(){
        return iterable != null;
    }

    @Override
    public boolean asyncSupported(){
        return true;
    }
    /**
     * Resets the iterator back to the beginning
     */
    @Override
    public void reset() {
        queue.clear();
        if (iterable != null)
            iterator = iterable.iterator();
    }

    /**
     * Batch size
     *
     * @return
     */
    @Override
    public int batch() {
        return batchSize;
    }

    /**
     * The current cursor if applicable
     *
     * @return
     */
    @Override
    public int cursor() {
        return 0;
    }

    /**
     * Total number of examples in the dataset
     *
     * @return
     */
    @Override
    public int numExamples() {
        return totalExamples();
    }

    /**
     * Set a pre processor
     *
     * @param preProcessor a pre processor to set
     */
    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        this.preProcessor = preProcessor;
    }

    /**
     * Get dataset iterator record reader labels
     */
    @Override
    public List<String> getLabels() {
        return labels;
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
        fillQueue();
        return !queue.isEmpty();
    }

    protected void fillQueue() {
        if (queue.isEmpty()) {
            INDArray labels = null;
            INDArray features = null;
            List<INDArray> cLabels = new ArrayList<>();
            List<INDArray> cFeatures = new ArrayList<>();

            for (int cnt = 0; cnt < batchSize; cnt++) {
                if (iterator.hasNext()) {
                        Pair<T, T> pair = iterator.next();
                        if (numFeatures < 1) {
                            if (pair.getFirst() instanceof INDArray) {
                                numFeatures = ((INDArray) pair.getFirst()).length();
                                numLabels = ((INDArray) pair.getSecond()).length();
                            } else if (pair.getFirst() instanceof float[]) {
                                numFeatures = ((float[]) pair.getFirst()).length;
                                numLabels = ((float[]) pair.getSecond()).length;
                            } else if (pair.getFirst() instanceof double[]) {
                                numFeatures = ((double[]) pair.getFirst()).length;
                                numLabels = ((double[]) pair.getSecond()).length;
                            }
                        }

                        if (pair.getFirst() instanceof INDArray) {
                            cFeatures.add(((INDArray) pair.getFirst()));
                            cLabels.add(((INDArray) pair.getSecond()));
                        } else if (pair.getFirst() instanceof float[]) {
                            cFeatures.add(Nd4j.create((float[]) pair.getFirst()));
                            cLabels.add(Nd4j.create((float[]) pair.getSecond()));
                        } else if (pair.getFirst() instanceof double[]) {
                            cFeatures.add(Nd4j.create((double[]) pair.getFirst()));
                            cLabels.add(Nd4j.create((double[]) pair.getSecond()));
                        }
                } else break;
            }

            if (cFeatures.size() == batchSize) {
                features = Nd4j.vstack(cFeatures);
                labels = Nd4j.vstack(cLabels);

                DataSet dataSet = new DataSet(features, labels);
                queue.add(dataSet);
            }
        }
    }

    /**
     * Returns the next element in the iteration.
     *
     * @return the next element in the iteration
     * @throws NoSuchElementException if the iteration has no more elements
     */
    @Override
    public DataSet next() throws NoSuchElementException {
        if (queue.isEmpty())
            throw new NoSuchElementException();

        DataSet dataSet = queue.poll();
        if (preProcessor != null)
            preProcessor.preProcess(dataSet);

        return dataSet;
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
     * @implSpec The default implementation throws an instance of
     * {@link UnsupportedOperationException} and performs no other action.
     */
    @Override
    public void remove() {
        // no-op
    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        return preProcessor;
    }
}
