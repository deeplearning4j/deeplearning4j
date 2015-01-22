package org.deeplearning4j.spark.util;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
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
     * From labeled point
     * @param sc the spark context used for creating the rdd
     * @param data the data to convert
     * @param numPossibleLabels the number of possible labels
     * @return
     */
    public static JavaRDD<DataSet> fromLabeledPoint(JavaSparkContext sc,JavaRDD<LabeledPoint> data,int numPossibleLabels) {
        List<DataSet> list  = fromLabeledPoint(data.collect(), numPossibleLabels);
        return sc.parallelize(list);
    }

    /**
     *
     * @param sc
     * @param data
     * @return
     */
    public static JavaRDD<LabeledPoint> fromDataSet(JavaSparkContext sc,JavaRDD<DataSet> data) {
        List<LabeledPoint> list  = toLabeledPoint(data.collect());
        return sc.parallelize(list);
    }


    /**
     *
     * @param labeledPoints
     * @return
     */
    private static List<LabeledPoint> toLabeledPoint(List<DataSet> labeledPoints) {
        List<LabeledPoint> ret = new ArrayList<>();
        for(DataSet point : labeledPoints)
            ret.add(toLabeledPoint(point));
        return ret;
    }

    /**
     *
     * @param point
     * @return
     */
    private static LabeledPoint toLabeledPoint(DataSet point) {
        Vector features = Vectors.dense(point.getFeatureMatrix().data().asDouble());
        double label = Nd4j.getBlasWrapper().iamax(point.getLabels());
        return new LabeledPoint(label,features);
    }


    /**
     *
     * @param labeledPoints
     * @param numPossibleLabels
     * @return
     */
    private static List<DataSet> fromLabeledPoint(List<LabeledPoint> labeledPoints,int numPossibleLabels) {
        List<DataSet> ret = new ArrayList<>();
        for(LabeledPoint point : labeledPoints)
            ret.add(fromLabeledPoint(point, numPossibleLabels));
        return ret;
    }

    /**
     *
     * @param point
     * @param numPossibleLabels
     * @return
     */
    private static DataSet fromLabeledPoint(LabeledPoint point,int numPossibleLabels) {
        Vector features = point.features();
        double label = point.label();
        return new DataSet(Nd4j.create(features.toArray()), FeatureUtil.toOutcomeVector((int) label, numPossibleLabels));
    }


}
