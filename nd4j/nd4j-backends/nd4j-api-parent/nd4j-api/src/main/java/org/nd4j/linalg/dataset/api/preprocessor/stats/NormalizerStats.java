/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.dataset.api.preprocessor.stats;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;

public interface NormalizerStats extends Serializable {
    interface Builder<S extends NormalizerStats> {
        Builder<S> addFeatures(org.nd4j.linalg.dataset.api.DataSet dataSet);

        /**
         * Add the labels of a DataSet to the statistics
         */
        Builder<S> addLabels(org.nd4j.linalg.dataset.api.DataSet dataSet);

        /**
         * Add rows of data to the statistics
         *
         * @param data the matrix containing multiple rows of data to include
         * @param mask (optionally) the mask of the data, useful for e.g. time series
         */
        Builder<S> add(INDArray data, INDArray mask);

        /**
         * DynamicCustomOpsBuilder pattern
         * @return
         */
        S build();
    }
}
