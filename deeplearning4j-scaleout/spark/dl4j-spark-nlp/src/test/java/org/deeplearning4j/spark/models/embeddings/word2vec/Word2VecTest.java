/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.deeplearning4j.spark.models.embeddings.word2vec;

import static org.junit.Assert.*;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.scaleout.perform.models.word2vec.Word2VecPerformer;
import org.deeplearning4j.spark.text.BaseSparkTest;
import org.deeplearning4j.spark.text.TextPipeline;
import org.deeplearning4j.text.sentenceiterator.CollectionSentenceIterator;
import org.junit.Test;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;
import org.springframework.core.io.ClassPathResource;

import java.util.Arrays;
import java.util.Collection;

/**
 * Created by agibsonccc on 1/31/15.
 */
public class Word2VecTest extends BaseSparkTest {

    @Test
    public void testSparkWord2Vec() throws Exception {
        JavaRDD<String> corpus = sc.textFile(new ClassPathResource("basic/word2vec.txt").getFile().getAbsolutePath());
        //test by rigging the rng to generate the same table
        Random random = Nd4j.getRandom();
        random.setSeed(123);
        org.deeplearning4j.models.word2vec.Word2Vec vec = new org.deeplearning4j.models.word2vec.Word2Vec.Builder()
                .iterate(new CollectionSentenceIterator(Arrays.asList("She found not one but two .")))
                .layerSize(100).minWordFrequency(1)
                .iterations(5).build();
        vec.fit();
        InMemoryLookupTable table = (InMemoryLookupTable) vec.lookupTable();

        random.setSeed(123);
        Word2Vec word2Vec = new Word2Vec();
        sc.getConf().set(Word2VecPerformer.NEGATIVE,String.valueOf(0));
        sc.getConf().set(TextPipeline.MIN_WORDS, String.valueOf("1"));
        Pair<VocabCache,WeightLookupTable> pair = word2Vec.train(corpus);
        InMemoryLookupTable table2 = (InMemoryLookupTable) pair.getSecond();
        assertEquals(vec.getVectorizer().vocab(),pair.getFirst());
        assertTrue(table.getSyn0().eps(table2.getSyn0()).sum(Integer.MAX_VALUE).getDouble(0) == table.getSyn0().length());
    }

    @Test
    public void testConcepts() throws Exception {
        JavaRDD<String> corpus = sc.textFile(new ClassPathResource("raw_sentences.txt").getFile().getAbsolutePath());
        Word2Vec word2Vec = new Word2Vec();
        sc.getConf().set(Word2VecPerformer.NEGATIVE, String.valueOf(0));
        Pair<VocabCache,WeightLookupTable> table = word2Vec.train(corpus);
        WordVectors vectors = WordVectorSerializer.fromPair(new Pair<>(table.getSecond(), table.getFirst()));
        Collection<String> words = vectors.wordsNearest("day", 20);
        assertTrue(words.contains("week"));
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
        SparkConf sparkConf = new SparkConf().set(org.deeplearning4j.spark.models.embeddings.word2vec.Word2VecPerformer.NUM_WORDS,"1")
                .setMaster("local[8]").set(Word2VecPerformer.NEGATIVE, String.valueOf(0)).set(TextPipeline.MIN_WORDS,String.valueOf("1"))
                .setAppName("sparktest");


        sc = new JavaSparkContext(sparkConf);
        return sc;

    }


}