package org.datavec.spark.transform.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

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
        if(records == null)
            records = new ArrayList<>();
        records.add(record);
    }


}
