package org.datavec.spark.transform.model;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.net.URI;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by kepricon on 17. 5. 24.
 */
@Data
@AllArgsConstructor
@NoArgsConstructor
public class BatchImageRecord {
    private List<SingleImageRecord> records;

    /**
     * Add a record
     * @param record
     */
    public void add(SingleImageRecord record) {
        if (records == null)
            records = new ArrayList<>();
        records.add(record);
    }

    public void add(URI uri) {
        this.add(new SingleImageRecord(uri));
    }
}
