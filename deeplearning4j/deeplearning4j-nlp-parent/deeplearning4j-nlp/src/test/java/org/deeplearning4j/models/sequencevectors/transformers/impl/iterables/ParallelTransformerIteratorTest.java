/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.models.sequencevectors.transformers.impl.iterables;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.transformers.impl.SentenceTransformer;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.text.documentiterator.BasicLabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelAwareIterator;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.MutipleEpochsSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.junit.Before;
import org.junit.Test;

import java.util.Iterator;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class ParallelTransformerIteratorTest {
    private TokenizerFactory factory = new DefaultTokenizerFactory();

    @Before
    public void setUp() throws Exception {

    }

    @Test
    public void hasNext() throws Exception {
        SentenceIterator iterator = new BasicLineIterator(new ClassPathResource("/big/raw_sentences.txt").getFile());

        SentenceTransformer transformer = new SentenceTransformer.Builder().iterator(iterator).allowMultithreading(true)
                        .tokenizerFactory(factory).build();

        Iterator<Sequence<VocabWord>> iter = transformer.iterator();
        int cnt = 0;
        Sequence<VocabWord> sequence = null;
        while (iter.hasNext()) {
            sequence = iter.next();
            assertNotEquals("Failed on [" + cnt + "] iteration", null, sequence);
            assertNotEquals("Failed on [" + cnt + "] iteration", 0, sequence.size());
            cnt++;
        }

        //   log.info("Last element: {}", sequence.asLabels());

        assertEquals(97162, cnt);
    }

    @Test
    public void testSpeedComparison1() throws Exception {
        SentenceIterator iterator = new MutipleEpochsSentenceIterator(
                        new BasicLineIterator(new ClassPathResource("/big/raw_sentences.txt").getFile()), 25);

        SentenceTransformer transformer = new SentenceTransformer.Builder().iterator(iterator)
                        .allowMultithreading(false).tokenizerFactory(factory).build();

        Iterator<Sequence<VocabWord>> iter = transformer.iterator();
        int cnt = 0;
        long time1 = System.currentTimeMillis();
        while (iter.hasNext()) {
            Sequence<VocabWord> sequence = iter.next();
            assertNotEquals("Failed on [" + cnt + "] iteration", null, sequence);
            assertNotEquals("Failed on [" + cnt + "] iteration", 0, sequence.size());
            cnt++;
        }
        long time2 = System.currentTimeMillis();

        log.info("Single-threaded time: {} ms", time2 - time1);
        iterator.reset();

        transformer = new SentenceTransformer.Builder().iterator(iterator).allowMultithreading(true)
                        .tokenizerFactory(factory).build();

        iter = transformer.iterator();

        time1 = System.currentTimeMillis();
        while (iter.hasNext()) {
            Sequence<VocabWord> sequence = iter.next();
            assertNotEquals("Failed on [" + cnt + "] iteration", null, sequence);
            assertNotEquals("Failed on [" + cnt + "] iteration", 0, sequence.size());
            cnt++;
        }
        time2 = System.currentTimeMillis();

        log.info("Multi-threaded time: {} ms", time2 - time1);


        SentenceIterator baseIterator = iterator;
        baseIterator.reset();


        LabelAwareIterator lai = new BasicLabelAwareIterator.Builder(new MutipleEpochsSentenceIterator(
                        new BasicLineIterator(new ClassPathResource("/big/raw_sentences.txt").getFile()), 25)).build();

        transformer = new SentenceTransformer.Builder().iterator(lai).allowMultithreading(false)
                        .tokenizerFactory(factory).build();

        iter = transformer.iterator();

        time1 = System.currentTimeMillis();
        while (iter.hasNext()) {
            Sequence<VocabWord> sequence = iter.next();
            assertNotEquals("Failed on [" + cnt + "] iteration", null, sequence);
            assertNotEquals("Failed on [" + cnt + "] iteration", 0, sequence.size());
            cnt++;
        }
        time2 = System.currentTimeMillis();

        log.info("Prefetched Single-threaded time: {} ms", time2 - time1);
        lai.reset();


        transformer = new SentenceTransformer.Builder().iterator(lai).allowMultithreading(true)
                        .tokenizerFactory(factory).build();

        iter = transformer.iterator();

        time1 = System.currentTimeMillis();
        while (iter.hasNext()) {
            Sequence<VocabWord> sequence = iter.next();
            assertNotEquals("Failed on [" + cnt + "] iteration", null, sequence);
            assertNotEquals("Failed on [" + cnt + "] iteration", 0, sequence.size());
            cnt++;
        }
        time2 = System.currentTimeMillis();

        log.info("Prefetched Multi-threaded time: {} ms", time2 - time1);

    }

}
