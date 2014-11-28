package org.deeplearning4j.scaleout.runner;

import org.deeplearning4j.scaleout.actor.TestPerformerFactory;
import org.deeplearning4j.scaleout.actor.runner.DeepLearning4jDistributed;
import org.deeplearning4j.scaleout.conf.Configuration;
import org.deeplearning4j.scaleout.job.Job;
import org.deeplearning4j.scaleout.job.JobIterator;
import org.deeplearning4j.scaleout.job.collection.CollectionJobIterator;
import org.deeplearning4j.scaleout.perform.WorkerPerformerFactory;
import org.deeplearning4j.scaleout.statetracker.StateTracker;
import org.deeplearning4j.scaleout.statetracker.hazelcast.HazelCastStateTracker;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.Set;

/**
 * Created by agibsonccc on 11/27/14.
 */
public class TestDistributed {

    private Collection<Job> testJobs;
    private StateTracker stateTracker;
    private JobIterator testIterator;
    private Configuration conf;


    @Before
    public void before() throws Exception {
        stateTracker = new HazelCastStateTracker();
        testJobs = new ArrayList<>();
        for(int i = 0; i < 10; i++) {
            testJobs.add(new Job("hello" + 1,""));
        }

        testIterator = new CollectionJobIterator(testJobs);
        conf = new Configuration();
        conf.set(WorkerPerformerFactory.WORKER_PERFORMER, TestPerformerFactory.class.getName());

    }

    @After
    public void finish() {
        testJobs.clear();
        stateTracker.finish();
    }

    @Test
    public void testDistributed() {
        DeepLearning4jDistributed distributed = new DeepLearning4jDistributed(testIterator,stateTracker);
        distributed.setup(conf);
        distributed.train();
        boolean done = false;
        int countDone = 0;
        Set<Job> remove = new HashSet<>();
        while(!done) {
            try {
                Thread.sleep(10000);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }

            for(Job j : testJobs) {
                if(j.getResult() != null && j.getResult().equals("done")) {
                    countDone++;
                    remove.add(j);
                }
            }

            testJobs.removeAll(remove);
            remove.clear();

            if(countDone >= 10)
                done = true;
        }




    }


}
