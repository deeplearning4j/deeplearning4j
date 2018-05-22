package org.deeplearning4j.spark.util.data.validation;

import org.apache.spark.api.java.function.Function2;
import org.deeplearning4j.spark.util.data.ValidationResult;

public class ValidationResultReduceFn implements Function2<ValidationResult, ValidationResult, ValidationResult> {
    @Override
    public ValidationResult call(ValidationResult v1, ValidationResult v2) throws Exception {
        return v1.add(v2);
    }
}
