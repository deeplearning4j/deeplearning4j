package org.deeplearning4j.spark.util;

import static org.junit.Assert.*;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.spark.BaseSparkTest;
import org.junit.Test;
import org.nd4j.linalg.dataset.DataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

/**
 * Created by agibsonccc on 1/23/15.
 */
public class MLLIbUtilTest extends BaseSparkTest {
    private static Logger log = LoggerFactory.getLogger(MLLIbUtilTest.class);

    @Test
    public void testMlLibTest() {
        DataSet dataSet = new IrisDataSetIterator(150,150).next();
        List<DataSet> list = dataSet.asList();
        JavaRDD<DataSet> data = sc.parallelize(list);
        JavaRDD<LabeledPoint> mllLibData = MLLibUtil.fromDataSet(sc,data);
        mllLibData.map(new AssertFunction());
    }

    public static class AssertFunction implements Function<LabeledPoint,Object> {

        @Override
        public Object call(LabeledPoint labeledPoint) throws Exception {
            assertEquals(150,labeledPoint.features().size());
            return null;
        }
    }


}
