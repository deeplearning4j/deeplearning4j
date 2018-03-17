package org.deeplearning4j.text.documentiterator;

import org.nd4j.linalg.io.ClassPathResource;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

/**
 * @author raver119@gmail.com
 */
public class AsyncLabelAwareIteratorTest {
    @Test
    public void nextDocument() throws Exception {
        SentenceIterator sentence = new BasicLineIterator(new ClassPathResource("/big/raw_sentences.txt").getFile());
        BasicLabelAwareIterator backed = new BasicLabelAwareIterator.Builder(sentence).build();

        int cnt = 0;
        while (backed.hasNextDocument()) {
            backed.nextDocument();
            cnt++;
        }
        assertEquals(97162, cnt);

        backed.reset();

        AsyncLabelAwareIterator iterator = new AsyncLabelAwareIterator(backed, 64);
        cnt = 0;
        while (iterator.hasNext()) {
            iterator.next();
            cnt++;

            if (cnt == 10)
                iterator.reset();
        }
        assertEquals(97172, cnt);
    }

}
