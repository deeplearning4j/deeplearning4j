package org.deeplearning4j.spark.util.data;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serializable;

/**
 * Result for validation of DataSet and MultiDataSets. See {@link SparkDataValidation} for more details
 *
 * @author Alex Black
 */
@AllArgsConstructor
@NoArgsConstructor
@Data
@Builder
public class ValidationResult implements Serializable {
    private long countTotal;
    private long countMissingFile;
    private long countTotalValid;
    private long countTotalInvalid;
    private long countLoadingFailure;
    private long countMissingFeatures;
    private long countMissingLabels;
    private long countInvalidFeatures;
    private long countInvalidLabels;
    private long countInvalidDeleted;

    public ValidationResult add(ValidationResult o){
        if(o == null){
            return this;
        }

        countTotal += o.countTotal;
        countMissingFile += o.countMissingFile;
        countTotalValid += o.countTotalValid;
        countTotalInvalid += o.countTotalInvalid;
        countLoadingFailure += o.countLoadingFailure;
        countMissingFeatures += o.countMissingFeatures;
        countMissingLabels += o.countMissingLabels;
        countInvalidFeatures += o.countInvalidFeatures;
        countInvalidLabels += o.countInvalidLabels;
        countInvalidDeleted += o.countInvalidDeleted;

        return this;
    }
}
