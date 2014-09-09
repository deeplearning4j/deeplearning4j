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
public class FileSentenceIteratorTest {

    private static Logger log = LoggerFactory.getLogger(FileSentenceIteratorTest.class);

    @Before
    public void before() throws Exception {
        File test = new File("dir");
        test.mkdir();
        File testFile = new File(test,"test.txt");
        FileUtils.writeLines(testFile, Arrays.asList("Hello", "My", "Name"));


        File multiDir = new File("multidir");
        for(int i = 0; i < 2; i++) {
            File newTestFile = new File(multiDir,"testfile-" + i);
            FileUtils.writeLines(newTestFile, Arrays.asList("Hello", "My", "Name"));

        }

    }


    @Test
    public void testUimaSentenceIterator() {

    }

    @Test
    public void testFileSentenceIterator() throws Exception {
        SentenceIterator iter =  new FileSentenceIterator(new File("dir"));
        assertTrue(iter.hasNext());

        String sentence = iter.nextSentence();
        assertTrue(iter.hasNext());
        assertEquals("Hello",sentence);
        assertEquals("My",iter.nextSentence());
        assertEquals("Name",iter.nextSentence());
        assertFalse(iter.hasNext());

        SentenceIterator multiIter =  new FileSentenceIterator(new File("dir"));
        assertTrue(multiIter.hasNext());
        for(int i = 0; i < 6; i++) {
            multiIter.nextSentence();
        }

        assertFalse(multiIter.hasNext());

    }

    @After
    public void after() throws Exception {
        File test = new File("dir");
        test.mkdir();
        FileUtils.deleteDirectory(test);
        FileUtils.deleteDirectory(new File("multidir"));
    }




}
