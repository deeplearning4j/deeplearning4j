package org.deeplearning4j.nearestneighbor.model;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.nd4j.linalg.dataset.DataSet;

import java.io.Serializable;

/**
 * Created by agibsonccc on 12/24/16.
 */
@Data
@AllArgsConstructor
@NoArgsConstructor
public class CSVRecord implements Serializable {
    private String[] values;

    /**
     * Instantiate a csv record from a vector
     * given either an input dataset and a
     * one hot matrix, the index will be appended to
     * the end of the record, or for regression
     * it will append all values in the labels
     * @param row the input vectors
     * @return the record from this {@link DataSet}
     */
    public static CSVRecord fromRow(DataSet row) {
        if (!row.getFeatureMatrix().isVector() && !row.getFeatureMatrix().isScalar())
            throw new IllegalArgumentException("Passed in dataset must represent a scalar or vector");
        if (!row.getLabels().isVector() && !row.getLabels().isScalar())
            throw new IllegalArgumentException("Passed in dataset labels must be a scalar or vector");
        //classification
        CSVRecord record;
        int idx = 0;
        if (row.getLabels().sumNumber().doubleValue() == 1.0) {
            String[] values = new String[row.getFeatureMatrix().columns() + 1];
            for (int i = 0; i < row.getFeatureMatrix().length(); i++) {
                values[idx++] = String.valueOf(row.getFeatureMatrix().getDouble(i));
            }
            int maxIdx = 0;
            for (int i = 0; i < row.getLabels().length(); i++) {
                if (row.getLabels().getDouble(maxIdx) < row.getLabels().getDouble(i)) {
                    maxIdx = i;
                }
            }

            values[idx++] = String.valueOf(maxIdx);
            record = new CSVRecord(values);
        }
        //regression (any number of values)
        else {
            String[] values = new String[row.getFeatureMatrix().columns() + row.getLabels().columns()];
            for (int i = 0; i < row.getFeatureMatrix().length(); i++) {
                values[idx++] = String.valueOf(row.getFeatureMatrix().getDouble(i));
            }
            for (int i = 0; i < row.getLabels().length(); i++) {
                values[idx++] = String.valueOf(row.getLabels().getDouble(i));
            }


            record = new CSVRecord(values);

        }
        return record;
    }

}
