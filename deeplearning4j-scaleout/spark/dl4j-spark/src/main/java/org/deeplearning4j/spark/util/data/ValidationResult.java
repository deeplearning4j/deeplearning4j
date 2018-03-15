package org.deeplearning4j.spark.util.data;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@AllArgsConstructor
@NoArgsConstructor
@Data
public class ValidationResult {
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
