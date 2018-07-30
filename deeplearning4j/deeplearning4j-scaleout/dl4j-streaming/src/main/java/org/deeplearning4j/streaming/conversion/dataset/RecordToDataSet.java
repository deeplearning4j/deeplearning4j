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

package org.deeplearning4j.streaming.conversion.dataset;

import org.datavec.api.writable.Writable;
import org.nd4j.linalg.dataset.DataSet;

import java.io.Serializable;
import java.util.Collection;

/**
 * Converts a list of records in to a dataset.
 * @author Adam Gibson
 */
public interface RecordToDataSet extends Serializable {

    /**
     * Converts records in to a dataset
     * @param records the records to convert
     * @param numLabels the number of labels for the dataset
     * @return the converted dataset.
     */
    DataSet convert(Collection<Collection<Writable>> records, int numLabels);

}
