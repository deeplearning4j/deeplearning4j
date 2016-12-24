package org.datavec.spark.transform.model;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serializable;

/**
 * Created by agibsonccc on 12/24/16.
 */
@Data
@AllArgsConstructor
@NoArgsConstructor
public class CSVRecord implements Serializable {
    private String[] values;
}
