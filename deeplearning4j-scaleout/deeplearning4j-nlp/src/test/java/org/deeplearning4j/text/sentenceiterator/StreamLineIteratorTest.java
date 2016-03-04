package org.deeplearning4j.text.sentenceiterator;

import org.canova.api.util.ClassPathResource;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileInputStream;

import static org.junit.Assert.*;

/**
 * Created by fartovii on 09.11.15.
 */
public class StreamLineIteratorTest {

    protected Logger logger = LoggerFactory.getLogger(StreamLineIteratorTest.class);

    @Test
    public void testHasNext() throws Exception {

        ClassPathResource reuters5250 = new ClassPathResource("/reuters/5250");
        File f = reuters5250.getFile();

        StreamLineIterator iterator = new StreamLineIterator.Builder(new FileInputStream(f))
                .setFetchSize(100)
                .build();

        int cnt = 0;
        while (iterator.hasNext()) {
            String line = iterator.nextSentence();

            assertNotEquals(null, line);
            logger.info("Line: " + line);
            cnt++;
        }

        assertEquals(24, cnt);
    }
}