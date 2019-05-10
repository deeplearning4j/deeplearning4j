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

import lombok.NonNull;
import org.nd4j.linalg.primitives.Pair;

/**
 * A simple utility iterator for creating a DataSetIterator from an {@code Iterable<Pair<double[], double[]>>}.
 * First value in pair is the features vector, second value in pair is the labels.
 * Supports generating 2d features/labels only
 *
 * @author raver119@gmail.com
 */
public class DoublesDataSetIterator extends AbstractDataSetIterator<double[]> {

    /**
     * @param iterable  Iterable to source data from
     * @param batchSize Batch size for generated DataSet objects
     */
    public DoublesDataSetIterator(@NonNull Iterable<Pair<double[], double[]>> iterable, int batchSize) {
        super(iterable, batchSize);
    }
}
