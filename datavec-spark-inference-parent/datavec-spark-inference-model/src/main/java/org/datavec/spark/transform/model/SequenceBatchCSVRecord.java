package org.datavec.spark.transform.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.MultiDataSet;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
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
     * Get the records as a list of strings directly
     * (this basically "unpacks" the objects)
     * @return
     */
    public List<List<String>> getRecordsAsString() {
        if(records == null)
            Collections.emptyList();
        List<List<String>> ret = new ArrayList<>(records.size());
        for(BatchCSVRecord record : records) {
            for(SingleCSVRecord singleCSVRecord : record.getRecords()) {
                ret.add(singleCSVRecord.getValues());
            }
        }

        return ret;
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
