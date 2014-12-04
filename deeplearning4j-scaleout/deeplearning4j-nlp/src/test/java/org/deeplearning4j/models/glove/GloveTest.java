package org.deeplearning4j.models.glove;

import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.UimaSentenceIterator;
import org.junit.Before;
import org.junit.Test;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import java.io.File;

/**
 * Created by agibsonccc on 12/3/14.
 */
public class GloveTest {
    private static Logger log = LoggerFactory.getLogger(GloveTest.class);
    private Glove glove;
    private SentenceIterator iter;

    @Before
    public void before() throws Exception {
        ClassPathResource resource = new ClassPathResource("/basic2/line2.txt");
        File file = resource.getFile().getParentFile();
        iter = UimaSentenceIterator.createWithPath(file.getAbsolutePath());


    }


    @Test
    public void testGlove() {
       glove = new Glove.Builder().iterate(iter).build();
       glove.fit();
    }


}
