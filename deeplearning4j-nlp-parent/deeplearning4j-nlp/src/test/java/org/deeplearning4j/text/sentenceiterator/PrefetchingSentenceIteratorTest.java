package org.deeplearning4j.text.sentenceiterator;

import org.nd4j.linalg.io.ClassPathResource;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * @author raver119@gmail.com
 */
public class PrefetchingSentenceIteratorTest {

    protected static final Logger log = LoggerFactory.getLogger(PrefetchingSentenceIteratorTest.class);

    @Test
    public void testHasMoreLinesFile() throws Exception {
        ClassPathResource resource = new ClassPathResource("/big/raw_sentences.txt");
        File file = resource.getFile();
        BasicLineIterator iterator = new BasicLineIterator(file);

        PrefetchingSentenceIterator fetcher =
                        new PrefetchingSentenceIterator.Builder(iterator).setFetchSize(1000).build();

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

        PrefetchingSentenceIterator fetcher =
                        new PrefetchingSentenceIterator.Builder(iterator).setFetchSize(1000).build();

        log.info("Phase 1 starting");

        int cnt = 0;
        while (fetcher.hasNext()) {
            String line = fetcher.nextSentence();
            // we'll imitate some workload in current thread by using ThreadSleep.
            // there's no need to keep it enabled forever, just uncomment next line if you're going to test this iterator.
            // otherwise this test will

            //    Thread.sleep(0, 10);

            cnt++;
            if (cnt % 10000 == 0)
                log.info("Line processed: " + cnt);
        }
    }

    @Test
    public void testPerformance1() throws Exception {
        ClassPathResource resource = new ClassPathResource("/big/raw_sentences.txt");
        File file = resource.getFile();

        BasicLineIterator iterator = new BasicLineIterator(file);

        PrefetchingSentenceIterator fetcher = new PrefetchingSentenceIterator.Builder(new BasicLineIterator(file))
                        .setFetchSize(500000).build();

        long time01 = System.currentTimeMillis();
        int cnt0 = 0;
        while (iterator.hasNext()) {
            iterator.nextSentence();
            cnt0++;
        }
        long time02 = System.currentTimeMillis();

        long time11 = System.currentTimeMillis();
        int cnt1 = 0;
        while (fetcher.hasNext()) {
            fetcher.nextSentence();
            cnt1++;
        }
        long time12 = System.currentTimeMillis();

        log.info("Basic iterator: " + (time02 - time01));

        log.info("Prefetched iterator: " + (time12 - time11));

        long difference = (time12 - time11) - (time02 - time01);
        log.info("Difference: " + difference);

        // on small corpus time difference can fluctuate a lot
        // but it's still can be used as effectiveness measurement
        assertTrue(difference < 150);
    }
}
