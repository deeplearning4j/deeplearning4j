package org.deeplearning4j.spark.util;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Matrices;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.FeatureUtil;

import java.util.ArrayList;
import java.util.List;


/**
 * Dl4j <----> MLLib
 *
 * @author Adam Gibson
 */
public class MLLibUtil {

    /**
     * Convert an ndarray to a matrix.
     * Note that the matrix will be con
     * @param arr the array
     * @return an mllib vector
     */
    public static INDArray toMatrix(Matrix arr) {
      return Nd4j.create(arr.toArray(), new int[]{arr.numRows(), arr.numCols()});
    }

    /**
     * Convert an ndarray to a vector
     * @param arr the array
     * @return an mllib vector
     */
    public static INDArray toVector(Vector arr) {
      return Nd4j.create(Nd4j.createBuffer(arr.toArray()));
    }


    /**
     * Convert an ndarray to a matrix.
     * Note that the matrix will be con
     * @param arr the array
     * @return an mllib vector
     */
    public static Matrix toMatrix(INDArray arr) {
      if(!arr.isMatrix()) {
        throw new IllegalArgumentException("passed in array must be a matrix");
      }
      return Matrices.dense(arr.rows(),arr.columns(),arr.data().asDouble());
    }

    /**
     * Convert an ndarray to a vector
     * @param arr the array
     * @return an mllib vector
     */
    public static Vector toVector(INDArray arr) {
      if(!arr.isVector()) {
        throw new IllegalArgumentException("passed in array must be a vector");
      }
      return Vectors.dense(arr.data().asDouble());
    }

    /**
     * From labeled point
     * @param sc the org.deeplearning4j.spark context used for creating the rdd
     * @param data the data to convert
     * @param numPossibleLabels the number of possible labels
     * @return
     */
    public static JavaRDD<DataSet> fromLabeledPoint(JavaSparkContext sc,JavaRDD<LabeledPoint> data,int numPossibleLabels) {
      List<DataSet> list  = fromLabeledPoint(data.collect(), numPossibleLabels);
      return sc.parallelize(list);
    }

    /**
     * Convert an rdd of data set in to labeled point
     * @param sc the spark context to use
     * @param data the dataset to convert
     * @return an rdd of labeled point
     */
    public static JavaRDD<LabeledPoint> fromDataSet(JavaSparkContext sc,JavaRDD<DataSet> data) {
        List<LabeledPoint> list  = toLabeledPoint(data.collect());
        return sc.parallelize(list);
    }


    /**
     * Convert a list of dataset in to a list of labeled points
     * @param labeledPoints the labeled points to convert
     * @return the labeled point list
     */
    private static List<LabeledPoint> toLabeledPoint(List<DataSet> labeledPoints) {
      List<LabeledPoint> ret = new ArrayList<>();
      for(DataSet point : labeledPoints) {
        ret.add(toLabeledPoint(point));
      }
      return ret;
    }

    /**
     * Convert a dataset (feature vector) to a labeled point
     * @param point the point to convert
     * @return the labeled point derived from this dataset
     */
    private static LabeledPoint toLabeledPoint(DataSet point) {
      if(!point.getFeatureMatrix().isVector()) {
        throw new IllegalArgumentException("Feature matrix must be a vector");
      }

      Vector features = toVector(point.getFeatureMatrix().dup());

      double label = Nd4j.getBlasWrapper().iamax(point.getLabels());
      return new LabeledPoint(label,features);
    }


    /**
     *
     * @param labeledPoints
     * @param numPossibleLabels
     * @return List of {@link DataSet}
     */
    private static List<DataSet> fromLabeledPoint(List<LabeledPoint> labeledPoints,int numPossibleLabels) {
      List<DataSet> ret = new ArrayList<>();
      for(LabeledPoint point : labeledPoints) {
        ret.add(fromLabeledPoint(point, numPossibleLabels));
      }
      return ret;
    }

    /**
     *
     * @param point
     * @param numPossibleLabels
     * @return {@link DataSet}
     */
    private static DataSet fromLabeledPoint(LabeledPoint point,int numPossibleLabels) {
      Vector features = point.features();
      double label = point.label();
      return new DataSet(Nd4j.create(features.toArray()), FeatureUtil.toOutcomeVector((int) label, numPossibleLabels));
    }


}
