package org.deeplearning4j.spark.text;

import static org.junit.Assert.*;

import org.apache.spark.api.java.JavaRDD;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.junit.Test;
import org.springframework.core.io.ClassPathResource;

import java.util.Collection;

/**
 * Created by agibsonccc on 1/29/15.
 */
public class TextPipelineTest extends BaseSparkTest {

    @Test
    public void testTextPipeline() throws Exception {
        JavaRDD<String> corpus = sc.textFile(new ClassPathResource("basic/word2vec.txt").getFile().getAbsolutePath());
        TextPipeline pipeline = new TextPipeline(corpus,1);
        Pair<VocabCache,Long> pair = pipeline.process();
        assertEquals(pair.getFirst().numWords(), 2);
        assertTrue(pair.getSecond() > 0);
        VocabCache vocab = pair.getFirst();
        Collection<String> words = vocab.words();
        assertTrue(words.contains("UNK") && words.contains("test"));
    }





}
