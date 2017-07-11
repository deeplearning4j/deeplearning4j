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
