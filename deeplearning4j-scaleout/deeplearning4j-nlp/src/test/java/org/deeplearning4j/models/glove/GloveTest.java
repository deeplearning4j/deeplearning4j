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

package org.deeplearning4j.models.glove;
import static org.junit.Assert.*;

import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import java.io.File;
import java.util.Collection;

/**
 * Created by agibsonccc on 12/3/14.
 */
public class GloveTest {
    private static final Logger log = LoggerFactory.getLogger(GloveTest.class);
    private Glove glove;
    private SentenceIterator iter;

    @Before
    public void before() throws Exception {

        ClassPathResource resource = new ClassPathResource("/raw_sentences.txt");
        File file = resource.getFile();
        iter = new LineSentenceIterator(file);
        iter.setPreProcessor(new SentencePreProcessor() {
            @Override
            public String preProcess(String sentence) {
                return sentence.toLowerCase();
            }
        });

    }


    @Ignore
    @Test
    public void testGlove() throws Exception {
        /*
        glove = new Glove.Builder().iterate(iter).symmetric(true).shuffle(true)
                .minWordFrequency(1).iterations(10).learningRate(0.1)
                .layerSize(300)
                .build();

        glove.fit();
        Collection<String> words = glove.wordsNearest("day", 20);
        log.info("Nearest words to 'day': " + words);
        assertTrue(words.contains("week"));

*/


    }


}
