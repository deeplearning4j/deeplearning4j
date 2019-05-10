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

package org.datavec.spark.transform.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.datavec.api.writable.Writable;
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
public class BatchCSVRecord implements Serializable {
    private List<SingleCSVRecord> records;


    /**
     * Get the records as a list of strings
     * (basically the underlying values for
     * {@link SingleCSVRecord})
     * @return
     */
    public List<List<String>> getRecordsAsString() {
        if(records == null)
            records = new ArrayList<>();
        List<List<String>> ret = new ArrayList<>();
        for(SingleCSVRecord csvRecord : records) {
            ret.add(csvRecord.getValues());
        }
        return ret;
    }


    /**
     * Create a batch csv record
     * from a list of writables.
     * @param batch
     * @return
     */
    public static BatchCSVRecord fromWritables(List<List<Writable>> batch) {
        List <SingleCSVRecord> records = new ArrayList<>(batch.size());
        for(List<Writable> list : batch) {
            List<String> add = new ArrayList<>(list.size());
            for(Writable writable : list) {
                add.add(writable.toString());
            }
            records.add(new SingleCSVRecord(add));
        }

        return BatchCSVRecord.builder().records(records).build();
    }


    /**
     * Add a record
     * @param record
     */
    public void add(SingleCSVRecord record) {
        if (records == null)
            records = new ArrayList<>();
        records.add(record);
    }


    /**
     * Return a batch record based on a dataset
     * @param dataSet the dataset to get the batch record for
     * @return the batch record
     */
    public static BatchCSVRecord fromDataSet(DataSet dataSet) {
        BatchCSVRecord batchCSVRecord = new BatchCSVRecord();
        for (int i = 0; i < dataSet.numExamples(); i++) {
            batchCSVRecord.add(SingleCSVRecord.fromRow(dataSet.get(i)));
        }

        return batchCSVRecord;
    }

}
