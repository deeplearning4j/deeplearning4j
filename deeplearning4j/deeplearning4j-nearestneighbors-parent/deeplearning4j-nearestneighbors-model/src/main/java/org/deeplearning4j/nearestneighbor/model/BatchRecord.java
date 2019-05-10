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

package org.deeplearning4j.nearestneighbor.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.nd4j.linalg.dataset.DataSet;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by agibsonccc on 1/21/17.
 */
@Data
@AllArgsConstructor
@Builder
@NoArgsConstructor
public class BatchRecord implements Serializable {
    private List<CSVRecord> records;

    /**
     * Add a record
     * @param record
     */
    public void add(CSVRecord record) {
        if (records == null)
            records = new ArrayList<>();
        records.add(record);
    }


    /**
     * Return a batch record based on a dataset
     * @param dataSet the dataset to get the batch record for
     * @return the batch record
     */
    public static BatchRecord fromDataSet(DataSet dataSet) {
        BatchRecord batchRecord = new BatchRecord();
        for (int i = 0; i < dataSet.numExamples(); i++) {
            batchRecord.add(CSVRecord.fromRow(dataSet.get(i)));
        }

        return batchRecord;
    }

}
