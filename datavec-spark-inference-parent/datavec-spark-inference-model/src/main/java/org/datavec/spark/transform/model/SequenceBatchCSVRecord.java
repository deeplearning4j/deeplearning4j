package org.datavec.spark.transform.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.MultiDataSet;

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
public class SequenceBatchCSVRecord implements Serializable {
    private List<BatchCSVRecord> records;

    /**
     * Add a record
     * @param record
     */
    public void add(BatchCSVRecord record) {
        if (records == null)
            records = new ArrayList<>();
        records.add(record);
    }


    /**
     * Return a batch record based on a dataset
     * @param dataSet the dataset to get the batch record for
     * @return the batch record
     */
    public static SequenceBatchCSVRecord fromDataSet(MultiDataSet dataSet) {
        SequenceBatchCSVRecord batchCSVRecord = new SequenceBatchCSVRecord();
        for (int i = 0; i < dataSet.numFeatureArrays(); i++) {
            batchCSVRecord.add(BatchCSVRecord.fromDataSet(new DataSet(dataSet.getFeatures(i),dataSet.getLabels(i))));
        }

        return batchCSVRecord;
    }

}
