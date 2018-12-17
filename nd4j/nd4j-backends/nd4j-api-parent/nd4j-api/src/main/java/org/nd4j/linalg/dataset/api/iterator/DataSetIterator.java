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

package org.nd4j.linalg.dataset.api.iterator;

import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;

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
public interface DataSetIterator extends IDataSetIterator<DataSet, DataSetPreProcessor> {

    /**
     * Input columns for the dataset
     *
     * @return
     */
    @Deprecated
    int inputColumns();

    /**
     * The number of labels for the dataset
     *
     * @return
     */
    @Deprecated
    int totalOutcomes();



    /**
     * Batch size
     *
     * @return
     */
    @Deprecated
    int batch();


    /**
     * Get dataset iterator class labels, if any.
     * Note that implementations are not required to implement this, and can simply return null
     */
    @Deprecated
    List<String> getLabels();

}
