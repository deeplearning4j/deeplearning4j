/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.scaleout.runner;

import org.canova.api.conf.Configuration;
import org.deeplearning4j.scaleout.actor.TestPerformerFactory;
import org.deeplearning4j.scaleout.actor.runner.DeepLearning4jDistributed;
import org.deeplearning4j.scaleout.job.Job;
import org.deeplearning4j.scaleout.job.JobIterator;
import org.deeplearning4j.scaleout.job.collection.CollectionJobIterator;
import org.deeplearning4j.scaleout.perform.WorkerPerformerFactory;
import org.deeplearning4j.scaleout.api.statetracker.StateTracker;
import org.deeplearning4j.scaleout.statetracker.hazelcast.HazelCastStateTracker;
import org.deeplearning4j.util.SerializationUtils;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import static org.junit.Assume.*;

import java.io.File;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;

/**
 * Created by agibsonccc on 11/27/14.
 */
public class TestDistributed {

    private Collection<Job> testJobs;
    private StateTracker stateTracker;
    private JobIterator testIterator;
    private Configuration conf;
    private File finalFile = new File("model-saver");

    @Before
    public void before() throws Exception {
        stateTracker = new HazelCastStateTracker();
        testJobs = new ArrayList<>();
        for(int i = 0; i < 10; i++) {
            testJobs.add(new Job("hello" + i,""));
        }

        testIterator = new CollectionJobIterator(testJobs);
        conf = new Configuration();
        conf.set(WorkerPerformerFactory.WORKER_PERFORMER, TestPerformerFactory.class.getName());

    }

    @After
    public void finish() throws Exception {
        testJobs.clear();
        stateTracker.finish();
        stateTracker.shutdown();

    }

    @Test
    public void testDistributed() throws Exception {
        DeepLearning4jDistributed distributed = new DeepLearning4jDistributed(testIterator,stateTracker);
        distributed.setup(conf);
        distributed.train();
        distributed.shutdown();



    }

    @Test
    public void testNextOne() {
        if(finalFile.exists()) {
            Serializable s = SerializationUtils.readObject(finalFile);
            assumeNotNull(s);
            finalFile.delete();
        }

    }

}
