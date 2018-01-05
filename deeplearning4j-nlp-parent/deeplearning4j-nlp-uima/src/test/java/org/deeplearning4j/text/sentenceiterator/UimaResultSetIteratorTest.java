package org.deeplearning4j.text.sentenceiterator;

import org.junit.Before;
import org.junit.Test;

import java.sql.ResultSet;

import static org.junit.Assert.assertEquals;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * @author Brad Heap nzv8fan@gmail.com
 */
public class UimaResultSetIteratorTest {

    @Before
    public void setUp() throws Exception {

    }

    @Test
    public void testSingleSentenceRow() throws Exception {

        // Setup a mock ResultSet object
        ResultSet resultSetMock = mock(ResultSet.class);

        // when .next() is called, first time true, then false
        when(resultSetMock.next()).thenReturn(true).thenReturn(false);
        when(resultSetMock.getString("line")).thenReturn("The quick brown fox.");

        UimaResultSetIterator iterator = new UimaResultSetIterator(resultSetMock, "line");

        int cnt = 0;
        while (iterator.hasNext()) {
            String line = iterator.nextSentence();
            cnt++;
        }

        assertEquals(1, cnt);

    }

    @Test
    public void testMultipleSentenceRow() throws Exception {

        // Setup a mock ResultSet object
        ResultSet resultSetMock = mock(ResultSet.class);

        // when .next() is called, first time true, then false
        when(resultSetMock.next()).thenReturn(true).thenReturn(false);
        when(resultSetMock.getString("line")).thenReturn("The quick brown fox. The lazy dog. Over a fence.");

        UimaResultSetIterator iterator = new UimaResultSetIterator(resultSetMock, "line");

        int cnt = 0;
        while (iterator.hasNext()) {
            String line = iterator.nextSentence();
            cnt++;
        }

        assertEquals(3, cnt);

    }

    @Test
    public void testMultipleSentencesAndMultipleRows() throws Exception {

        // Setup a mock ResultSet object
        ResultSet resultSetMock = mock(ResultSet.class);

        // when .next() is called, first time true, then false
        when(resultSetMock.next()).thenReturn(true).thenReturn(true).thenReturn(false);
        when(resultSetMock.getString("line")).thenReturn("The quick brown fox.")
                        .thenReturn("The lazy dog. Over a fence.");

        UimaResultSetIterator iterator = new UimaResultSetIterator(resultSetMock, "line");

        int cnt = 0;
        while (iterator.hasNext()) {
            String line = iterator.nextSentence();
            cnt++;
        }

        assertEquals(3, cnt);

    }

    @Test
    public void testMultipleSentencesAndMultipleRowsAndReset() throws Exception {

        // Setup a mock ResultSet object
        ResultSet resultSetMock = mock(ResultSet.class);

        // when .next() is called, first time true, then false
        when(resultSetMock.next()).thenReturn(true).thenReturn(true).thenReturn(false).thenReturn(true).thenReturn(true)
                        .thenReturn(false);
        when(resultSetMock.getString("line")).thenReturn("The quick brown fox.")
                        .thenReturn("The lazy dog. Over a fence.").thenReturn("The quick brown fox.")
                        .thenReturn("The lazy dog. Over a fence.");

        UimaResultSetIterator iterator = new UimaResultSetIterator(resultSetMock, "line");

        int cnt = 0;
        while (iterator.hasNext()) {
            String line = iterator.nextSentence();
            cnt++;
        }

        assertEquals(3, cnt);

        iterator.reset();

        cnt = 0;
        while (iterator.hasNext()) {
            String line = iterator.nextSentence();
            cnt++;
        }

        assertEquals(3, cnt);
    }

}
