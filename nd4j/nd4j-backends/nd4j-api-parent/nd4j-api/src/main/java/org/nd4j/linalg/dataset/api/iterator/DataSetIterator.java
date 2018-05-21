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
 *
 */

package org.nd4j.linalg.dataset.api.iterator;

import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;

import java.io.Serializable;
import java.util.Iterator;
import java.util.List;


/**
 * A DataSetIterator handles
 * traversing through a dataset and preparing
 * <p/>
 * data for a neural network.
 * <p/>
 * Typical usage of an iterator is akin to:
 * <p/>
 * DataSetIterator iter = ..;
 * <p/>
 * while(iter.hasNext()) {
 * DataSet d = iter.next();
 * //iterate network...
 * }
 * <p/>
 * <p/>
 * For custom numbers of examples/batch sizes you can call:
 * <p/>
 * iter.next(num)
 * <p/>
 * where num is the number of examples to fetch
 *
 * @author Adam Gibson
 */
public interface DataSetIterator extends Iterator<DataSet>, Serializable {

    /**
     * Like the standard next method but allows a
     * customizable number of examples returned
     *
     * @param num the number of examples
     * @return the next data applyTransformToDestination
     */
    DataSet next(int num);

    /**
     * Total examples in the iterator
     *
     * @return
     */
    int totalExamples();

    /**
     * Input columns for the dataset
     *
     * @return
     */
    int inputColumns();

    /**
     * The number of labels for the dataset
     *
     * @return
     */
    int totalOutcomes();


    /**
     * Is resetting supported by this DataSetIterator? Many DataSetIterators do support resetting,
     * but some don't
     *
     * @return true if reset method is supported; false otherwise
     */
    boolean resetSupported();

    /**
     * Does this DataSetIterator support asynchronous prefetching of multiple DataSet objects?
     * Most DataSetIterators do, but in some cases it may not make sense to wrap this iterator in an
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
    boolean asyncSupported();

    /**
     * Resets the iterator back to the beginning
     */
    void reset();

    /**
     * Batch size
     *
     * @return
     */
    int batch();

    /**
     * The current cursor if applicable
     *
     * @return
     */
    int cursor();

    /**
     * Total number of examples in the dataset
     *
     * @return
     */
    int numExamples();


    /**
     * Set a pre processor
     *
     * @param preProcessor a pre processor to set
     */
    void setPreProcessor(DataSetPreProcessor preProcessor);

    /**
     * Returns preprocessors, if defined
     *
     * @return
     */
    DataSetPreProcessor getPreProcessor();

    /**
     * Get dataset iterator record reader labels
     */
    List<String> getLabels();

}
