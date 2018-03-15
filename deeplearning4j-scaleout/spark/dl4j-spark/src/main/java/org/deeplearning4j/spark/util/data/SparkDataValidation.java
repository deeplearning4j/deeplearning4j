package org.deeplearning4j.spark.util.data;

import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.spark.util.SparkUtils;
import org.deeplearning4j.spark.util.data.validation.ValidationResultReduceFn;
import org.deeplearning4j.spark.util.data.validation.dataset.ValidateDataSetFn;

import java.io.IOException;

public class SparkDataValidation {

    private SparkDataValidation(){ }

    public ValidationResult validateDataSets(SparkContext sc, String path, boolean recursive, boolean deleteInvalid, int[] featuresShape, int[] labelsShape){
        return validateDataSets(new JavaSparkContext(sc), path, recursive, deleteInvalid, featuresShape, labelsShape);
    }

    public ValidationResult validateDataSets(JavaSparkContext sc, String path, boolean recursive, boolean deleteInvalid, int[] featuresShape, int[] labelsShape){
        JavaRDD<String> paths;
        try {
            paths = SparkUtils.listPaths(sc, path, recursive);
        } catch (IOException e) {
            throw new RuntimeException("Error listing paths in directory", e);
        }

        JavaRDD<ValidationResult> results = paths.map(new ValidateDataSetFn(deleteInvalid, featuresShape, labelsShape));

        return results.reduce(new ValidationResultReduceFn());
    }




}
