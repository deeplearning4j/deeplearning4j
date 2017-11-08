package org.deeplearning4j.text.documentiterator;


import lombok.extern.slf4j.Slf4j;
import org.datavec.api.util.ClassPathResource;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.InputStream;

import static org.junit.Assert.assertEquals;

/**
 * Created by fartovii on 09.11.15.
 */

@Ignore
@Slf4j
public class FileDocumentIteratorTest {

    @Before
    public void setUp() throws Exception {

    }

    /**
     * Checks actual number of documents retrieved by DocumentIterator
     * @throws Exception
     */
    @Test
    public void testNextDocument() throws Exception {
        ClassPathResource reuters5250 = new ClassPathResource("/reuters/5250");
        File f = reuters5250.getFile();

        DocumentIterator iter = new FileDocumentIterator(f.getAbsolutePath());

        log.info(f.getAbsolutePath());

        int cnt = 0;
        while (iter.hasNext()) {
            InputStream stream = iter.nextDocument();
            stream.close();
            cnt++;
        }

        assertEquals(24, cnt);
    }


    /**
     * Checks actual number of documents retrieved by DocumentIterator after being RESET
     * @throws Exception
     */
    @Test
    public void testDocumentReset() throws Exception {
        ClassPathResource reuters5250 = new ClassPathResource("/reuters/5250");
        File f = reuters5250.getFile();

        DocumentIterator iter = new FileDocumentIterator(f.getAbsolutePath());

        int cnt = 0;
        while (iter.hasNext()) {
            InputStream stream = iter.nextDocument();
            stream.close();
            cnt++;
        }

        iter.reset();

        while (iter.hasNext()) {
            InputStream stream = iter.nextDocument();
            stream.close();
            cnt++;
        }

        assertEquals(48, cnt);
    }
}
