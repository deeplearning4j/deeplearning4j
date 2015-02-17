package org.deeplearning4j.models.paragraphvectors;


import org.deeplearning4j.text.sentenceiterator.labelaware.LabelAwareSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.labelaware.LabelAwareUimaSentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.UimaTokenizerFactory;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import java.io.File;
import java.util.ArrayList;

import static org.junit.Assert.assertEquals;

/**
 * Created by agibsonccc on 12/3/14.
 */
public class ParagraphVectorsTest {
    private static final Logger log = LoggerFactory.getLogger(ParagraphVectorsTest.class);

    @Before
    public void before() {
        new File("word2vec-index").delete();
    }

    @Test
    public void testWord2VecRunThroughVectors() throws Exception {
        ClassPathResource resource = new ClassPathResource("/basic2/line2.txt");
        File file = resource.getFile().getParentFile();
        LabelAwareSentenceIterator iter = LabelAwareUimaSentenceIterator.createWithPath(file.getAbsolutePath());
        new File("cache.ser").delete();


        TokenizerFactory t = new UimaTokenizerFactory();


        ParagraphVectors vec = new ParagraphVectors.Builder()
                .minWordFrequency(1).iterations(5)
                .layerSize(100)
                .stopWords(new ArrayList<String>())
                .windowSize(5).iterate(iter).tokenizerFactory(t).build();

        assertEquals(new ArrayList<String>(), vec.getStopWords());


        vec.fit();
        double sim = vec.similarity("Adam","deeplearning4j");
        new File("cache.ser").delete();

    }


}
