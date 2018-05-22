package org.deeplearning4j.text.documentiterator;

import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 * Created by raver on 26.11.2015.
 */
public class LabelsSourceTest {

    @Before
    public void setUp() throws Exception {

    }

    @Test
    public void testNextLabel1() throws Exception {
        LabelsSource generator = new LabelsSource("SENTENCE_");

        assertEquals("SENTENCE_0", generator.nextLabel());
    }

    @Test
    public void testNextLabel2() throws Exception {
        LabelsSource generator = new LabelsSource("SENTENCE_%d_HAHA");

        assertEquals("SENTENCE_0_HAHA", generator.nextLabel());
    }

    @Test
    public void testNextLabel3() throws Exception {
        List<String> list = Arrays.asList("LABEL0", "LABEL1", "LABEL2");
        LabelsSource generator = new LabelsSource(list);

        assertEquals("LABEL0", generator.nextLabel());
    }

    @Test
    public void testLabelsCount1() throws Exception {
        List<String> list = Arrays.asList("LABEL0", "LABEL1", "LABEL2");
        LabelsSource generator = new LabelsSource(list);

        assertEquals("LABEL0", generator.nextLabel());
        assertEquals("LABEL1", generator.nextLabel());
        assertEquals("LABEL2", generator.nextLabel());

        assertEquals(3, generator.getNumberOfLabelsUsed());
    }

    @Test
    public void testLabelsCount2() throws Exception {
        LabelsSource generator = new LabelsSource("SENTENCE_");

        assertEquals("SENTENCE_0", generator.nextLabel());
        assertEquals("SENTENCE_1", generator.nextLabel());
        assertEquals("SENTENCE_2", generator.nextLabel());
        assertEquals("SENTENCE_3", generator.nextLabel());
        assertEquals("SENTENCE_4", generator.nextLabel());

        assertEquals(5, generator.getNumberOfLabelsUsed());
    }

    @Test
    public void testLabelsCount3() throws Exception {
        LabelsSource generator = new LabelsSource("SENTENCE_");

        assertEquals("SENTENCE_0", generator.nextLabel());
        assertEquals("SENTENCE_1", generator.nextLabel());
        assertEquals("SENTENCE_2", generator.nextLabel());
        assertEquals("SENTENCE_3", generator.nextLabel());
        assertEquals("SENTENCE_4", generator.nextLabel());

        assertEquals(5, generator.getNumberOfLabelsUsed());

        generator.reset();

        assertEquals(5, generator.getNumberOfLabelsUsed());
    }
}
