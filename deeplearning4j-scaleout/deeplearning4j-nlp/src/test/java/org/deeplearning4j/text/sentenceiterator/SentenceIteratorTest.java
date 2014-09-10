package org.deeplearning4j.text.sentenceiterator;

import static org.junit.Assert.*;

import org.apache.commons.io.FileUtils;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Arrays;

/**
 * Created by agibsonccc on 9/9/14.
 */
public class SentenceIteratorTest {

    private static Logger log = LoggerFactory.getLogger(SentenceIteratorTest.class);

    @Before
    public void before() throws Exception {
        File test = new File("dir");
        test.mkdir();
        File testFile = new File(test,"test.txt");
        FileUtils.writeLines(testFile, Arrays.asList("Hello", "My", "Name"));


        File multiDir = new File("multidir");
        for(int i = 0; i < 2; i++) {
            File newTestFile = new File(multiDir,"testfile-" + i);
            FileUtils.writeLines(newTestFile, Arrays.asList("Sentence 1.","Sentence 2.","Sentence 3."));

        }

    }


    @Test
    public void testUimaSentenceIterator() throws Exception {
        SentenceIterator multiIter = UimaSentenceIterator.createWithPath("multidir");
        SentenceIterator iter = UimaSentenceIterator.createWithPath("dir");
        testMulti(multiIter,1);

    }

    @Test
    public void testFileSentenceIterator() throws Exception {
        SentenceIterator iter =  new FileSentenceIterator(new File("dir"));
        SentenceIterator multiIter = new FileSentenceIterator(new File("multidir"));
        testSingle(iter);
        testMulti(multiIter,3);

    }


    public void testSingle(SentenceIterator iter) {
        assertTrue(iter.hasNext());

        String sentence = iter.nextSentence();
        assertTrue(iter.hasNext());
        assertEquals("Hello",sentence);
        assertEquals("My",iter.nextSentence());
        assertEquals("Name",iter.nextSentence());
        assertFalse(iter.hasNext());

    }

    public void testMulti(SentenceIterator iter,int expectedSentences) {
        assertTrue(iter.hasNext());
        for(int i = 0; i < expectedSentences * 2; i++) {
            iter.nextSentence();
        }

        assertFalse(iter.hasNext());

    }

    @After
    public void after() throws Exception {
        File test = new File("dir");
        test.mkdir();
        FileUtils.deleteDirectory(test);
        FileUtils.deleteDirectory(new File("multidir"));
    }




}
