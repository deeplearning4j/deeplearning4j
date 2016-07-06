package org.canova.spark;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.junit.After;
import org.junit.Before;

import java.io.Serializable;

public abstract class BaseSparkTest  implements Serializable {
    protected transient JavaSparkContext sc;

    @Before
    public void before() {
        sc = getContext();
    }

    @After
    public void after() {
        sc.close();
        sc = null;
    }

    public JavaSparkContext getContext() {
        if(sc != null)
            return sc;
        // set to test mode
        SparkConf sparkConf = new SparkConf()
                .setMaster("local[*]")
                .setAppName("sparktest");


        sc = new JavaSparkContext(sparkConf);

        return sc;
    }
}
