/*-
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

package org.deeplearning4j.text.sentenceiterator;

import org.apache.commons.io.FileUtils;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Arrays;

import static org.junit.Assert.*;

/**
 * Created by agibsonccc on 9/9/14.
 */
public class SentenceIteratorTest {

    private static final Logger log = LoggerFactory.getLogger(SentenceIteratorTest.class);

    @Before
    public void before() throws Exception {
        File test = new File("dir");
        test.mkdir();
        File testFile = new File(test, "test.txt");
        FileUtils.writeLines(testFile, Arrays.asList("Hello", "My", "Name"));


        File multiDir = new File("multidir");
        for (int i = 0; i < 2; i++) {
            File newTestFile = new File(multiDir, "testfile-" + i);
            FileUtils.writeLines(newTestFile, Arrays.asList("Sentence 1.", "Sentence 2.", "Sentence 3."));

        }

    }


    @Test
    public void testUimaSentenceIterator() throws Exception {
        SentenceIterator multiIter = UimaSentenceIterator.createWithPath("multidir");
        SentenceIterator iter = UimaSentenceIterator.createWithPath("dir");
        testMulti(multiIter, 1);

    }

    @Test
    public void testFileSentenceIterator() throws Exception {
        SentenceIterator iter = new FileSentenceIterator(new File("dir"));
        SentenceIterator multiIter = new FileSentenceIterator(new File("multidir"));
        testSingle(iter);
        testMulti(multiIter, 3);

    }



    public void testSingle(SentenceIterator iter) {
        assertTrue(iter.hasNext());

        String sentence = iter.nextSentence();
        assertTrue(iter.hasNext());
        assertEquals("Hello", sentence);
        assertEquals("My", iter.nextSentence());
        assertEquals("Name", iter.nextSentence());
        assertFalse(iter.hasNext());

    }

    public void testMulti(SentenceIterator iter, int expectedSentences) {
        assertTrue(iter.hasNext());
        for (int i = 0; i < expectedSentences * 2; i++) {
            iter.nextSentence();
        }

        assertFalse(iter.hasNext());

    }

    @After
    public void after() throws Exception {
        File test = new File("dir");
        test.mkdir();
        FileUtils.deleteQuietly(test);
        FileUtils.deleteQuietly(new File("multidir"));
    }



}
