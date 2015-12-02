package org.deeplearning4j.text.sentenceiterator;

import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import java.io.File;

import static org.junit.Assert.*;

/**
 * Created by fartovii on 28.11.15.
 */
public class PrefetchingSentenceIteratorTest {

    protected static final Logger log = LoggerFactory.getLogger(PrefetchingSentenceIteratorTest.class);

    @Test
    public void testHasMoreLinesFile() throws Exception {
        ClassPathResource resource = new ClassPathResource("/big/raw_sentences.txt");
        File file = resource.getFile();
        BasicLineIterator iterator = new BasicLineIterator(file);

        PrefetchingSentenceIterator fetcher = new PrefetchingSentenceIterator.Builder(iterator)
                .build();

        log.info("Phase 1 starting");

        int cnt = 0;
        while (fetcher.hasNext()) {
            String line = fetcher.nextSentence();
//            log.info(line);
            cnt++;
        }


        assertEquals(97162, cnt);

        log.info("Phase 2 starting");
        fetcher.reset();

        cnt = 0;
        while (fetcher.hasNext()) {
            String line = fetcher.nextSentence();
            cnt++;
        }

        assertEquals(97162, cnt);
    }
}