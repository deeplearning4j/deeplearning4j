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

package org.deeplearning4j.datasets.iterator;

import lombok.AllArgsConstructor;
import lombok.NoArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;

/**
 * This dataset iterator combines multiple DataSetIterators into 1 MultiDataSetIterator.
 * Values from each iterator are joined on a per-example basis - i.e., the values from each DataSet are combined
 * as different feature arrays for a multi-input neural network.
 * Labels can come from either one of the underlying DataSetIteartors only (if 'outcome' is >= 0) or from all
 * iterators (if outcome is < 0)
 *
 * @author raver119@gmail.com
 */
@Slf4j
@NoArgsConstructor
@AllArgsConstructor
public class JointMultiDataSetIterator implements MultiDataSetIterator {
    protected MultiDataSetPreProcessor preProcessor;
    protected Collection<DataSetIterator> iterators;
    protected int outcome = -1;

    /**
     * @param iterators Underlying iterators to wrap
     */
    public JointMultiDataSetIterator(DataSetIterator... iterators) {
        this.iterators = new ArrayList<DataSetIterator>();
        this.iterators.addAll(Arrays.asList(iterators));
        this.outcome = -1;
    }

    /**
     *
     * @param outcome   Index to get the label from. If < 0, labels from all iterators will be used to create the
     *                  final MultiDataSet
     * @param iterators Underlying iterators to wrap
     */
    public JointMultiDataSetIterator(int outcome, DataSetIterator... iterators){
        this(iterators);
        this.outcome = outcome;
    }

    /**
     * Fetch the next 'num' examples. Similar to the next method, but returns a specified number of examples
     *
     * @param num Number of examples to fetch
     */
    @Override
    public MultiDataSet next(int num) {
        throw new UnsupportedOperationException();
    }

    /**
     * Set the preprocessor to be applied to each MultiDataSet, before each MultiDataSet is returned.
     *
     * @param preProcessor MultiDataSetPreProcessor. May be null.
     */
    @Override
    public void setPreProcessor(MultiDataSetPreProcessor preProcessor) {
        this.preProcessor = preProcessor;
    }

    /**
     * Get the {@link MultiDataSetPreProcessor}, if one has previously been set.
     * Returns null if no preprocessor has been set
     *
     * @return Preprocessor
     */
    @Override
    public MultiDataSetPreProcessor getPreProcessor() {
        return preProcessor;
    }

    /**
     * Is resetting supported by this DataSetIterator? Many DataSetIterators do support resetting,
     * but some don't
     *
     * @return true if reset method is supported; false otherwise
     */
    @Override
    public boolean resetSupported() {
        boolean sup = true;

        for (val i: iterators)
            if (!i.resetSupported()) {
                sup = false;
                break;
            }

        return sup;
    }

    /**
     * Does this MultiDataSetIterator support asynchronous prefetching of multiple MultiDataSet objects?
     * Most MultiDataSetIterators do, but in some cases it may not make sense to wrap this iterator in an
     * iterator that does asynchronous prefetching. For example, it would not make sense to use asynchronous
     * prefetching for the following types of iterators:
     * (a) Iterators that store their full contents in memory already
     * (b) Iterators that re-use features/labels arrays (as future next() calls will overwrite past contents)
     * (c) Iterators that already implement some level of asynchronous prefetching
     * (d) Iterators that may return different data depending on when the next() method is called
     *
     * @return true if asynchronous prefetching from this iterator is OK; false if asynchronous prefetching should not
     * be used with this iterator
     */
    @Override
    public boolean asyncSupported() {
        boolean sup = true;

        for (val i: iterators)
            if (!i.asyncSupported()) {
                sup = false;
                break;
            }

        return sup;
    }

    /**
     * Resets the iterator back to the beginning
     */
    @Override
    public void reset() {
        for (val i: iterators)
            i.reset();
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
        boolean has = true;

        for (val i: iterators)
            if (!i.hasNext()) {
                has = false;
                break;
            }

        return has;
    }

    /**
     * Returns the next element in the iteration.
     *
     * @return the next element in the iteration
     */
    @Override
    public MultiDataSet next() {
        val features = new ArrayList<INDArray>();
        val labels = new ArrayList<INDArray>();
        val featuresMask = new ArrayList<INDArray>();
        val labelsMask = new ArrayList<INDArray>();

        boolean hasFM = false;
        boolean hasLM = false;

        int cnt = 0;
        for (val i: iterators) {
            val ds = i.next();

            features.add(ds.getFeatures());
            featuresMask.add(ds.getFeaturesMaskArray());

            if (outcome < 0 || cnt == outcome) {
                labels.add(ds.getLabels());
                labelsMask.add(ds.getLabelsMaskArray());
            }

            if (ds.getFeaturesMaskArray() != null)
                hasFM = true;

            if (ds.getLabelsMaskArray() != null)
                hasLM = true;

            cnt++;
        }

        INDArray[] fm = hasFM ? featuresMask.toArray(new INDArray[0]) : null;
        INDArray[] lm = hasLM ? labelsMask.toArray(new INDArray[0]) : null;

        val mds = new org.nd4j.linalg.dataset.MultiDataSet(features.toArray(new INDArray[0]), labels.toArray(new INDArray[0]), fm, lm);

        if (preProcessor != null)
            preProcessor.preProcess(mds);

        return mds;
    }

    /**
     * PLEASE NOTE: This method is NOT implemented
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
        // noopp
    }
}
