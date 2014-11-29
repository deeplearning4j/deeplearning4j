package org.deeplearning4j.scaleout.perform.text;

import static org.junit.Assert.*;

import org.deeplearning4j.berkeley.Counter;
import org.deeplearning4j.scaleout.conf.Configuration;
import org.deeplearning4j.scaleout.job.Job;
import org.deeplearning4j.scaleout.perform.WorkerPerformer;
import org.junit.Test;

/**
 * Created by agibsonccc on 11/29/14.
 */
public class WordCountTest {
    @Test
    public void testWordCount() {
        WorkerPerformer performer = new WordCountWorkPerformer();
        Configuration conf = new Configuration();
        performer.setup(conf);
        Job j = new Job("This is one sentence.","");
        performer.perform(j);
        Counter<String> result = (Counter<String>) j.getResult();
        assertEquals(1.0,result.getCount("This"),1e-1);

    }



}
