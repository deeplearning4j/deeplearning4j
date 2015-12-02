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
                .setFetchSize(1000)
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

    @Test
    public void testLoadedIterator1() throws Exception {
        ClassPathResource resource = new ClassPathResource("/big/raw_sentences.txt");
        File file = resource.getFile();
        BasicLineIterator iterator = new BasicLineIterator(file);

        PrefetchingSentenceIterator fetcher = new PrefetchingSentenceIterator.Builder(iterator)
                .setFetchSize(1000)
                .build();

        log.info("Phase 1 starting");

        int cnt = 0;
        while (fetcher.hasNext()) {
            String line = fetcher.nextSentence();
            // we'll imitate some workload in current thread by using ThreadSleep.
            // there's no need to keep it enabled forever, just uncomment next line if you're going to test this iterator.
            // otherwise this test will

           //    Thread.sleep(0, 10);

            cnt++;
            if (cnt % 10000 == 0) log.info("Line processed: " + cnt);
        }
    }
}