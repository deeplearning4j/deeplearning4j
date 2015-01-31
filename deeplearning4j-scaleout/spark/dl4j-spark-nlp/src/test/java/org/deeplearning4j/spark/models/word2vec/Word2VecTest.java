package org.deeplearning4j.spark.models.word2vec;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.spark.text.BaseSparkTest;
import org.junit.Test;
import org.springframework.core.io.ClassPathResource;

/**
 * Created by agibsonccc on 1/31/15.
 */
public class Word2VecTest extends BaseSparkTest {

    @Test
    public void testSparkWord2Vec() throws Exception {
        JavaRDD<String> corpus = sc.textFile(new ClassPathResource("basic/word2vec.txt").getFile().getAbsolutePath());
        Word2Vec word2Vec = new Word2Vec();
        sc.getConf().set(Word2VecPerformer.NEGATIVE,String.valueOf(0));
        word2Vec.train(corpus);

    }


    /**
     *
     * @return
     */
    @Override
    public JavaSparkContext getContext() {
        if(sc != null)
            return sc;
        // set to test mode
        SparkConf sparkConf = new SparkConf()
                .setMaster("local[8]").set(Word2VecPerformer.NEGATIVE, String.valueOf(0))
                .setAppName("sparktest");


        sc = new JavaSparkContext(sparkConf);
        return sc;

    }


}
