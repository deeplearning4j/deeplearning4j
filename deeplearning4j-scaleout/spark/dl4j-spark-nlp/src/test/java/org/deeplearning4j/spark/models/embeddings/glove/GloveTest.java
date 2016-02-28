/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.spark.models.embeddings.glove;

import static org.junit.Assert.assertTrue;

import java.util.Collection;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.canova.api.util.ClassPathResource;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.glove.GloveWeightLookupTable;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.spark.text.BaseSparkTest;
import org.junit.Ignore;
import org.junit.Test;

/**
 * Created by agibsonccc on 1/31/15.
 */
@Ignore
public class GloveTest extends BaseSparkTest {

    @Test
    public void testGlove() throws Exception {
        Glove glove = new Glove(true,5,100);
        JavaRDD<String> corpus = sc.textFile(new ClassPathResource("raw_sentences.txt").getFile().getAbsolutePath()).map(new Function<String, String>() {
            @Override
            public String call(String s) throws Exception {
                return s.toLowerCase();
            }
        }).cache();


        Pair<VocabCache<VocabWord>,GloveWeightLookupTable> table = glove.train(corpus);
        WordVectors vectors = WordVectorSerializer.fromPair(new Pair<>((InMemoryLookupTable) table.getSecond(), (VocabCache) table.getFirst()));
        Collection<String> words = vectors.wordsNearest("day", 20);
        assertTrue(words.contains("week"));
    }

}
