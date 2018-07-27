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
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.FeatureUtil;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * Assumes csv format and converts a batch of records in to a
 * size() x record length matrix.
 *
 * @author Adam Gibson
 */
public class CSVRecordToDataSet implements RecordToDataSet {
    @Override
    public DataSet convert(Collection<Collection<Writable>> records, int numLabels) {
        //all but last label
        DataSet ret = new DataSet(Nd4j.create(records.size(), records.iterator().next().size() - 1),
                        Nd4j.create(records.size(), numLabels));
        //  INDArray ret = Nd4j.create(records.size(),records.iterator().next().size() - 1);
        int count = 0;
        for (Collection<Writable> record : records) {
            List<Writable> list;
            if (record instanceof List) {
                list = (List<Writable>) record;
            } else
                list = new ArrayList<>(record);
            DataSet d = new DataSet(Nd4j.create(record.size() - 1),
                            FeatureUtil.toOutcomeVector(list.get(list.size() - 1).toInt(), numLabels));
            ret.addRow(d, count++);

        }


        return ret;
    }
}
