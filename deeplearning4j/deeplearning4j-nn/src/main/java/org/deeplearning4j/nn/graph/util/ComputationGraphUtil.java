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

package org.deeplearning4j.nn.graph.util;

import org.deeplearning4j.datasets.iterator.impl.MultiDataSetIteratorAdapter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

public class ComputationGraphUtil {

    private ComputationGraphUtil() {}

    /** Convert a DataSet to the equivalent MultiDataSet */
    public static MultiDataSet toMultiDataSet(DataSet dataSet) {
        INDArray f = dataSet.getFeatures();
        INDArray l = dataSet.getLabels();
        INDArray fMask = dataSet.getFeaturesMaskArray();
        INDArray lMask = dataSet.getLabelsMaskArray();

        INDArray[] fNew = f == null ? null : new INDArray[] {f};
        INDArray[] lNew = l == null ? null : new INDArray[] {l};
        INDArray[] fMaskNew = (fMask != null ? new INDArray[] {fMask} : null);
        INDArray[] lMaskNew = (lMask != null ? new INDArray[] {lMask} : null);

        return new org.nd4j.linalg.dataset.MultiDataSet(fNew, lNew, fMaskNew, lMaskNew);
    }

    /** Convert a DataSetIterator to a MultiDataSetIterator, via an adaptor class */
    public static MultiDataSetIterator toMultiDataSetIterator(DataSetIterator iterator) {
        return new MultiDataSetIteratorAdapter(iterator);
    }

}
