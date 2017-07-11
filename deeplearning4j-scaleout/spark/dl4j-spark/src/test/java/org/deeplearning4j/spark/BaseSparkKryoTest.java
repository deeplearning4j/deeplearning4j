package org.deeplearning4j.spark;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;

/**
 * Created by Alex on 04/07/2017.
 */
public class BaseSparkKryoTest extends BaseSparkTest {

    @Override
    public JavaSparkContext getContext() {
        if (sc != null) {
            return sc;
        }

        SparkConf sparkConf = new SparkConf().setMaster("local[" + numExecutors() + "]").setAppName("sparktest");

        sparkConf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer");
        sparkConf.set("spark.kryo.registrator", "org.nd4j.Nd4jRegistrator");

        sc = new JavaSparkContext(sparkConf);

        return sc;
    }

}

