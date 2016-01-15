package org.deeplearning4j.text.documentiterator;

import org.canova.api.util.ClassPathResource;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.*;

/**
 * @author raver119@gmail.com
 */
public class FilenamesLabelAwareIteratorTest {

    @Before
    public void setUp() throws Exception {

    }

    @Test
    public void testNextDocument() throws Exception {
        FilenamesLabelAwareIterator iterator = new FilenamesLabelAwareIterator.Builder()
                .addSourceFolder(new ClassPathResource("/big").getFile())
                .useAbsolutePathAsLabel(false)
                .build();

        List<String> labels = new ArrayList<>();

        LabelledDocument doc1 = iterator.nextDocument();
        labels.add(doc1.getLabel());

        LabelledDocument doc2 = iterator.nextDocument();
        labels.add(doc2.getLabel());

        LabelledDocument doc3 = iterator.nextDocument();
        labels.add(doc3.getLabel());

        LabelledDocument doc4 = iterator.nextDocument();
        labels.add(doc4.getLabel());

        assertFalse(iterator.hasNextDocument());

        System.out.println("Labels: " + labels);

        assertTrue(labels.contains("coc.txt"));
        assertTrue(labels.contains("occurrences.txt"));
        assertTrue(labels.contains("raw_sentences.txt"));
        assertTrue(labels.contains("tokens.txt"));
    }
}