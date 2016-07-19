package org.nd4j;

import lombok.AllArgsConstructor;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.broadcast.Broadcast;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 * Created by Alex on 04/07/2016.
 */
public class TestNd4jKryoSerialization {

    private JavaSparkContext sc;

    @Before
    public void before(){
        SparkConf sparkConf = new SparkConf();
        sparkConf.setMaster("local[*]");
        sparkConf.setAppName("Iris");

        sparkConf.set("spark.serializer","org.apache.spark.serializer.KryoSerializer");
        sparkConf.set("spark.kryo.registrator", "org.nd4j.Nd4jRegistrator");

        sc = new JavaSparkContext(sparkConf);
    }

    @Test
    public void testSerialization(){

        Tuple2<INDArray,INDArray> t2 = new Tuple2<>(
                Nd4j.linspace(1,10,10),
                Nd4j.linspace(10,20,10));

        Broadcast<Tuple2<INDArray,INDArray>> b = sc.broadcast(t2);

        List<INDArray> list = new ArrayList<>();
        for( int i=0; i<100; i++ ){
            list.add(Nd4j.ones(5));
        }

        JavaRDD<INDArray> rdd = sc.parallelize(list);

        rdd.foreach(new AssertFn(b));
    }


    @After
    public void after(){
        if(sc != null) sc.close();
    }

    @AllArgsConstructor
    public static class AssertFn implements VoidFunction<INDArray> {

        private Broadcast<Tuple2<INDArray,INDArray>> b;

        @Override
        public void call(INDArray arr) throws Exception {
            Tuple2<INDArray,INDArray> t2 = b.getValue();
            assertEquals(Nd4j.linspace(1,10,10), t2._1());
            assertEquals(Nd4j.linspace(10,20,10), t2._2());

            assertEquals(Nd4j.ones(5), arr);
        }
    }
}
