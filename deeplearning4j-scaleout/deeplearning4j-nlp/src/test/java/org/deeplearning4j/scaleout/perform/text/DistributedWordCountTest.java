package org.deeplearning4j.scaleout.perform.text;

import static org.junit.Assert.*;

import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.scaleout.aggregator.JobAggregator;
import org.deeplearning4j.scaleout.conf.Configuration;
import org.deeplearning4j.scaleout.job.JobIterator;
import org.deeplearning4j.scaleout.testsupport.BaseTestDistributed;
import org.deeplearning4j.text.sentenceiterator.FileSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.util.SerializationUtils;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.springframework.core.io.ClassPathResource;

import java.io.File;

/**
 * Created by agibsonccc on 11/29/14.
 */
public class DistributedWordCountTest extends BaseTestDistributed {

    private SentenceIterator iter;

    @Before
    public void before() throws Exception {
        File rootDir = new ClassPathResource("/rootdir").getFile();
        iter = new FileSentenceIterator(rootDir);
        init();
    }


    @Override
    public Configuration createConfiguration() {
        Configuration ret =  super.createConfiguration();
        ret.set(JobAggregator.AGGREGATOR,WordCountJobAggregator.class.getName());
        ret.set(WordCountJobAggregator.MIN_WORD_FREQUENCY,"1");
        return ret;
    }

    @After
    public void after() throws Exception {
        tearDown();
    }

    @Test
    public void testDistributedWordCount() {
        distributed.train();
        VocabCache cache = SerializationUtils.readObject(new File("model-saver"));
        assertEquals(2,cache.wordFrequency("file."));
    }


    @Override
    public String workPerformFactoryClassName() {
        return WordCountWorkPerformerFactory.class.getName();
    }

    @Override
    public JobIterator createIterator() {
        return new SentenceJobIterator(iter);
    }
}
