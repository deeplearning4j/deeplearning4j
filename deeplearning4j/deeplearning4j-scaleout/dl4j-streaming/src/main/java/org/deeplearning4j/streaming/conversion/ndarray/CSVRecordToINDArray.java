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

import org.datavec.api.util.ndarray.RecordConverter;
import org.datavec.api.writable.Writable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Collection;

/**
 * Assumes csv format and converts a batch of records in to a
 * size() x record length matrix.
 *
 * @author Adam Gibson
 */
public class CSVRecordToINDArray implements RecordToNDArray {
    @Override
    public INDArray convert(Collection<Collection<Writable>> records) {
        INDArray ret = Nd4j.create(records.size(), records.iterator().next().size());
        int count = 0;
        for (Collection<Writable> record : records) {
            ret.putRow(count++, RecordConverter.toArray(record));
        }
        return ret;
    }
}
