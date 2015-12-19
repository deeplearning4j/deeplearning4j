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
     * Get dataset iterator record reader labels
     *
     */
    List<String> getLabels();

}
