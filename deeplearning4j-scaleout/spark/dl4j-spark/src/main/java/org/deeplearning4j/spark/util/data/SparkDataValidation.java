package org.deeplearning4j.spark.util.data;

import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.spark.util.SparkUtils;
import org.deeplearning4j.spark.util.data.validation.ValidateMultiDataSetFn;
import org.deeplearning4j.spark.util.data.validation.ValidationResultReduceFn;
import org.deeplearning4j.spark.util.data.validation.ValidateDataSetFn;

import java.io.IOException;
import java.io.OutputStream;
import java.util.List;

/**
 * Utilities for validating DataSets and MultiDataSets saved (usually) in a HDFS directory.
 *
 * @author Alex Black
 */
public class SparkDataValidation {

    private SparkDataValidation() {
    }

    /**
     * Validate DataSet objects saved to the specified directory on HDFS by attempting to load them and checking their
     * contents. Assumes DataSets were saved using {@link org.nd4j.linalg.dataset.DataSet#save(OutputStream)}.<br>
     * Note: this method will also consider all files in subdirectories (i.e., is recursive).
     *
     * @param sc   Spark context
     * @param path HDFS path of the directory containing the saved DataSet objects
     * @return Results of the validation
     */
    public static ValidationResult validateDataSets(JavaSparkContext sc, String path) {
        return validateDataSets(sc, path, true, false, null, null);
    }

    /**
     * Validate DataSet objects saved to the specified directory on HDFS by attempting to load them and checking their
     * contents. Assumes DataSets were saved using {@link org.nd4j.linalg.dataset.DataSet#save(OutputStream)}.<br>
     * This method (optionally) additionally validates the arrays using the specified shapes for the features and labels.
     * Note: this method will also consider all files in subdirectories (i.e., is recursive).
     *
     * @param sc            Spark context
     * @param path          HDFS path of the directory containing the saved DataSet objects
     * @param featuresShape May be null. If non-null: feature arrays must match the specified shape, for all values with
     *                      shape > 0. For example, if featuresShape = {-1,10} then the features must be rank 2,
     *                      can have any size for the first dimension, but must have size 10 for the second dimension.
     * @param labelsShape   As per featuresShape, but for the labels instead
     * @return Results of the validation
     */
    public static ValidationResult validateDataSets(JavaSparkContext sc, String path, int[] featuresShape, int[] labelsShape) {
        return validateDataSets(sc, path, true, false, featuresShape, labelsShape);
    }

    /**
     * Validate DataSet objects - <b>and delete any invalid DataSets</b> - that have been previously saved to the
     * specified directory on HDFS by attempting to load them and checking their contents. Assumes DataSets were saved
     * using {@link org.nd4j.linalg.dataset.DataSet#save(OutputStream)}.<br>
     * Note: this method will also consider all files in subdirectories (i.e., is recursive).
     *
     * @param sc   Spark context
     * @param path HDFS path of the directory containing the saved DataSet objects
     * @return Results of the validation/deletion
     */
    public static ValidationResult deleteInvalidDataSets(JavaSparkContext sc, String path) {
        return validateDataSets(sc, path, true, true, null, null);
    }

    /**
     * Validate DataSet objects - <b>and delete any invalid DataSets</b> - that have been previously saved to the
     * specified directory on HDFS by attempting to load them and checking their contents. Assumes DataSets were saved
     * using {@link org.nd4j.linalg.dataset.DataSet#save(OutputStream)}.<br>
     * This method (optionally) additionally validates the arrays using the specified shapes for the features and labels.
     * Note: this method will also consider all files in subdirectories (i.e., is recursive).
     *
     * @param sc            Spark context
     * @param path          HDFS path of the directory containing the saved DataSet objects
     * @param featuresShape May be null. If non-null: feature arrays must match the specified shape, for all values with
     *                      shape > 0. For example, if featuresShape = {-1,10} then the features must be rank 2,
     *                      can have any size for the first dimension, but must have size 10 for the second dimension.
     * @param labelsShape   As per featuresShape, but for the labels instead
     * @return Results of the validation
     */
    public static ValidationResult deleteInvalidDataSets(JavaSparkContext sc, String path, int[] featuresShape, int[] labelsShape) {
        return validateDataSets(sc, path, true, true, featuresShape, labelsShape);
    }


    protected static ValidationResult validateDataSets(SparkContext sc, String path, boolean recursive, boolean deleteInvalid,
                                                int[] featuresShape, int[] labelsShape) {
        return validateDataSets(new JavaSparkContext(sc), path, recursive, deleteInvalid, featuresShape, labelsShape);
    }

    protected static ValidationResult validateDataSets(JavaSparkContext sc, String path, boolean recursive, boolean deleteInvalid,
                                                int[] featuresShape, int[] labelsShape) {
        JavaRDD<String> paths;
        try {
            paths = SparkUtils.listPaths(sc, path, recursive);
        } catch (IOException e) {
            throw new RuntimeException("Error listing paths in directory", e);
        }

        JavaRDD<ValidationResult> results = paths.map(new ValidateDataSetFn(deleteInvalid, featuresShape, labelsShape));

        return results.reduce(new ValidationResultReduceFn());
    }


    /**
     * Validate MultiDataSet objects saved to the specified directory on HDFS by attempting to load them and checking their
     * contents. Assumes MultiDataSets were saved using {@link org.nd4j.linalg.dataset.MultiDataSet#save(OutputStream)}.<br>
     * Note: this method will also consider all files in subdirectories (i.e., is recursive).
     *
     * @param sc   Spark context
     * @param path HDFS path of the directory containing the saved DataSet objects
     * @return Results of the validation
     */
    public static ValidationResult validateMultiDataSets(JavaSparkContext sc, String path) {
        return validateMultiDataSets(sc, path, true, false, -1, -1, null, null);
    }

    /**
     * Validate MultiDataSet objects saved to the specified directory on HDFS by attempting to load them and checking their
     * contents. Assumes MultiDataSets were saved using {@link org.nd4j.linalg.dataset.MultiDataSet#save(OutputStream)}.<br>
     * This method additionally validates that the expected number of feature/labels arrays are present in all MultiDataSet
     * objects<br>
     * Note: this method will also consider all files in subdirectories (i.e., is recursive).
     *
     * @param sc               Spark context
     * @param path             HDFS path of the directory containing the saved DataSet objects
     * @param numFeatureArrays Number of feature arrays that are expected for the MultiDataSet (set -1 to not check)
     * @param numLabelArrays   Number of labels arrays that are expected for the MultiDataSet (set -1 to not check)
     * @return Results of the validation
     */
    public static ValidationResult validateMultiDataSets(JavaSparkContext sc, String path, int numFeatureArrays, int numLabelArrays) {
        return validateMultiDataSets(sc, path, true, false, numFeatureArrays, numLabelArrays, null, null);
    }


    /**
     * Validate MultiDataSet objects saved to the specified directory on HDFS by attempting to load them and checking their
     * contents. Assumes MultiDataSets were saved using {@link org.nd4j.linalg.dataset.MultiDataSet#save(OutputStream)}.<br>
     * This method (optionally) additionally validates the arrays using the specified shapes for the features and labels.
     * Note: this method will also consider all files in subdirectories (i.e., is recursive).
     *
     * @param sc            Spark context
     * @param path          HDFS path of the directory containing the saved DataSet objects
     * @param featuresShape May be null. If non-null: feature arrays must match the specified shapes, for all values with
     *                      shape > 0. For example, if featuresShape = {{-1,10}} then there must be 1 features array,
     *                      features array 0 must be rank 2, can have any size for the first dimension, but must have
     *                      size 10 for the second dimension.
     * @param labelsShape   As per featuresShape, but for the labels instead
     * @return Results of the validation
     */
    public static ValidationResult validateMultiDataSets(JavaSparkContext sc, String path, List<int[]> featuresShape, List<int[]> labelsShape) {
        return validateMultiDataSets(sc, path, true, false, (featuresShape == null ? -1 : featuresShape.size()),
                (labelsShape == null ? -1 : labelsShape.size()), featuresShape, labelsShape);
    }

    /**
     * Validate MultiDataSet objects - <b>and delete any invalid MultiDataSets</b> - that have been previously saved to the
     * specified directory on HDFS by attempting to load them and checking their contents. Assumes MultiDataSets were saved
     * using {@link org.nd4j.linalg.dataset.MultiDataSet#save(OutputStream)}.<br>
     * Note: this method will also consider all files in subdirectories (i.e., is recursive).
     *
     * @param sc   Spark context
     * @param path HDFS path of the directory containing the saved DataSet objects
     * @return Results of the validation/deletion
     */
    public static ValidationResult deleteInvalidMultiDataSets(JavaSparkContext sc, String path) {
        return validateMultiDataSets(sc, path, true, true, -1, -1, null, null);
    }

    /**
     * Validate MultiDataSet objects - <b>and delete any invalid MultiDataSets</b> - that have been previously saved
     * to the specified directory on HDFS by attempting to load them and checking their contents. Assumes MultiDataSets
     * were saved using {@link org.nd4j.linalg.dataset.MultiDataSet#save(OutputStream)}.<br>
     * This method (optionally) additionally validates the arrays using the specified shapes for the features and labels,
     * Note: this method will also consider all files in subdirectories (i.e., is recursive).
     *
     * @param sc            Spark context
     * @param path          HDFS path of the directory containing the saved DataSet objects
     * @param featuresShape May be null. If non-null: feature arrays must match the specified shapes, for all values with
     *                      shape > 0. For example, if featuresShape = {{-1,10}} then there must be 1 features array,
     *                      features array 0 must be rank 2, can have any size for the first dimension, but must have
     *                      size 10 for the second dimension.
     * @param labelsShape   As per featuresShape, but for the labels instead
     * @return Results of the validation
     */
    public static ValidationResult deleteInvalidMultiDataSets(JavaSparkContext sc, String path, List<int[]> featuresShape,
                                                       List<int[]> labelsShape) {
        return validateMultiDataSets(sc, path, true, true, (featuresShape == null ? -1 : featuresShape.size()),
                (labelsShape == null ? -1 : labelsShape.size()), featuresShape, labelsShape);
    }

    /**
     * Validate MultiDataSet objects - <b>and delete any invalid MultiDataSets</b> - that have been previously saved
     * to the specified directory on HDFS by attempting to load them and checking their contents. Assumes MultiDataSets
     * were saved using {@link org.nd4j.linalg.dataset.MultiDataSet#save(OutputStream)}.<br>
     * This method (optionally) additionally validates the arrays using the specified shapes for the features and labels.
     * Note: this method will also consider all files in subdirectories (i.e., is recursive).
     *
     * @param sc               Spark context
     * @param path             HDFS path of the directory containing the saved DataSet objects
     * @param numFeatureArrays Number of feature arrays that are expected for the MultiDataSet (set -1 to not check)
     * @param numLabelArrays   Number of labels arrays that are expected for the MultiDataSet (set -1 to not check)
     * @return Results of the validation
     */
    public static ValidationResult deleteInvalidMultiDataSets(JavaSparkContext sc, String path, int numFeatureArrays, int numLabelArrays) {
        return validateMultiDataSets(sc, path, true, true, numFeatureArrays, numLabelArrays, null, null);
    }

    protected static ValidationResult validateMultiDataSets(SparkContext sc, String path, boolean recursive, boolean deleteInvalid,
                                                     int numFeatureArrays, int numLabelArrays,
                                                     List<int[]> featuresShape, List<int[]> labelsShape) {
        return validateMultiDataSets(new JavaSparkContext(sc), path, recursive, deleteInvalid, numFeatureArrays, numLabelArrays,
                featuresShape, labelsShape);
    }

    protected static ValidationResult validateMultiDataSets(JavaSparkContext sc, String path, boolean recursive, boolean deleteInvalid,
                                                     int numFeatureArrays, int numLabelArrays,
                                                     List<int[]> featuresShape, List<int[]> labelsShape) {
        JavaRDD<String> paths;
        try {
            paths = SparkUtils.listPaths(sc, path, recursive);
        } catch (IOException e) {
            throw new RuntimeException("Error listing paths in directory", e);
        }

        JavaRDD<ValidationResult> results = paths.map(new ValidateMultiDataSetFn(deleteInvalid, numFeatureArrays, numLabelArrays,
                featuresShape, labelsShape));

        return results.reduce(new ValidationResultReduceFn());
    }


}
