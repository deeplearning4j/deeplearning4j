package org.nd4j.etl4j.spark.transform.utils;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;

/**
 * Factory to create spark connections
 */
public class SparkConnectFactory {

    public static String name = "SparkConnectFactory";

    public static SparkConf config() {
        return config(name);
    }

    public static SparkConf config(String name){
        SparkConf sparkConf = new SparkConf();
        sparkConf.setMaster("local[*]");
        sparkConf.set("spark.driver.maxResultSize", "2G");
        sparkConf.setAppName(name);
        return sparkConf;
    }

    public static JavaSparkContext getContext(){
        return getContext(name);
    }

    public static JavaSparkContext getContext(String name){
        return new JavaSparkContext(config(name));
    }

}