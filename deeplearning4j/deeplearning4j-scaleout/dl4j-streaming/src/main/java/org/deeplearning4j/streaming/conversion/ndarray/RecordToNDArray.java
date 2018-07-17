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

package org.deeplearning4j.streaming.conversion.ndarray;

import org.datavec.api.writable.Writable;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;
import java.util.Collection;

/**
 * A function convert from record to ndarrays
 * @author Adam Gibson
 */
public interface RecordToNDArray extends Serializable {

    /**
     * Converts a list of records in to 1 ndarray
     * @param records the records to convert
     * @return the collection of records
     * to convert
     */
    INDArray convert(Collection<Collection<Writable>> records);

}
