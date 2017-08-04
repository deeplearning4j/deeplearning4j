package org.deeplearning4j.streaming.pipeline.spark;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.streaming.api.java.JavaDStream;

/**
 * In order to handle changes between Spark 1.x and 2.x
 */
public class StreamingContextUtils {

    public static <K> void foreach(JavaDStream<K> stream, VoidFunction<JavaRDD<K>> func) {
        stream.foreachRDD(func);
    }
}
