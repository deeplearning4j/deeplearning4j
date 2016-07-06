package org.canova.spark.transform;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;

/**
 * Created by Alex on 1/06/2016.
 */
public class BaseSparkTest {

    public static JavaSparkContext sc;

    @BeforeClass
    public static void beforeClass(){
        SparkConf conf = new SparkConf();
        conf.setAppName("Test");
        conf.setMaster("local[*]");
        sc = new JavaSparkContext(conf);
    }

    @AfterClass
    public static void afterClass(){
        sc.stop();
    }

}
