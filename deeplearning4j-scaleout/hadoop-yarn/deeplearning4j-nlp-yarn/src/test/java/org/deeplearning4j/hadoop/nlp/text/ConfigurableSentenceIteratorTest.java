package org.deeplearning4j.hadoop.nlp.text;

import static org.junit.Assert.*;

import org.apache.commons.io.FileUtils;
import org.apache.hadoop.conf.Configuration;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.File;
import java.io.IOException;

/**
 * Created by agibsonccc on 1/29/15.
 */
public class ConfigurableSentenceIteratorTest {
    private Configuration conf;

    @Before
    public void before() throws IOException {
        conf = new Configuration();
        conf.set("fs.defaultFS", "file:///");
        File parentDir = new File("parent");
        parentDir.mkdir();
        FileUtils.writeStringToFile(new File(parentDir,"touch"),"hello");
        conf.set(ConfigurableSentenceIterator.ROOT_PATH,parentDir.toURI().toString());


    }

    @After
    public void after() throws IOException {
        FileUtils.deleteDirectory(new File("parent"));
    }

    @Test
    public void testSentenceIterator() throws IOException {
        TestConfigurableSentenceIterator iter = new TestConfigurableSentenceIterator(conf);
        assertEquals(true,iter.hasNext());
        String next = iter.nextSentence();
        assertEquals("hello",next);
        assertFalse(iter.hasNext());

    }

}
