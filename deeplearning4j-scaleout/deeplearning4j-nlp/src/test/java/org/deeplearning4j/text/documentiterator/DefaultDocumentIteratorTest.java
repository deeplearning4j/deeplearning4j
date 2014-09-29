package org.deeplearning4j.text.documentiterator;
import static org.junit.Assert.*;

import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.junit.Test;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import java.io.File;
import java.io.InputStream;

/**
 * Created by agibsonccc on 9/29/14.
 */
public class DefaultDocumentIteratorTest {

    private static Logger log = LoggerFactory.getLogger(DefaultDocumentIteratorTest.class);
    @Test
    public void testDocumentIterator() throws Exception {
        ClassPathResource reuters5250 = new ClassPathResource("/reuters/5250");
        File f = reuters5250.getFile();

        DocumentIterator iter = new FileDocumentIterator(f.getAbsolutePath());

        InputStream doc = iter.nextDocument();

        TokenizerFactory t = new DefaultTokenizerFactory();
        Tokenizer next = t.create(doc);
        String[] list = "PEARSON CONCENTRATES ON FOUR SECTORS".split(" ");
        ///PEARSON CONCENTRATES ON FOUR SECTORS
        int count = 0;
        while(next.hasMoreTokens() && count < list.length) {
            String token = next.nextToken();
            assertEquals(list[count++],token);
        }


        doc.close();


    }

}
