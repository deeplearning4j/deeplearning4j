package org.deeplearning4j.spark;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.junit.After;
import org.junit.Before;

/**
 * Created by agibsonccc on 1/23/15.
 */
public abstract class BaseSparkTest {
    protected JavaSparkContext sc;

    @Before
    public void before() {
        sc = getContext();
    }
    @After
    public void after() {
        sc.close();
        sc = null;
    }

    /**
     *
     * @return
     */
    public JavaSparkContext getContext() {
        if(sc != null)
            return sc;
        // set to test mode
        SparkConf sparkConf = new SparkConf()
                .setMaster("local[8]")
                .setAppName("sparktest");


        sc = new JavaSparkContext(sparkConf);
        return sc;

    }

}
