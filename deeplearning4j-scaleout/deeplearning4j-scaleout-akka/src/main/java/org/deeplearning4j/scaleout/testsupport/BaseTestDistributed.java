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

package org.deeplearning4j.scaleout.testsupport;

import org.canova.api.conf.Configuration;
import org.deeplearning4j.scaleout.actor.runner.DeepLearning4jDistributed;
import org.deeplearning4j.scaleout.job.JobIterator;
import org.deeplearning4j.scaleout.perform.WorkerPerformerFactory;
import org.deeplearning4j.scaleout.api.statetracker.StateTracker;
import org.deeplearning4j.scaleout.statetracker.hazelcast.HazelCastStateTracker;

import java.io.File;

/**
 * Baseline test support for testing a distributed app
 * @author Adam Gibson
 */
public abstract class BaseTestDistributed {
    protected JobIterator iterator;
    protected DeepLearning4jDistributed distributed;
    protected StateTracker stateTracker;
    protected String workPerformFactoryClassName;
    private Configuration conf;

    public void init() throws Exception {
        if(stateTracker != null) {
            stateTracker.finish();
            stateTracker.shutdown();
        }
        stateTracker = createStateTracker();
        iterator = createIterator();
        workPerformFactoryClassName = workPerformFactoryClassName();
        conf = createConfiguration();
        distributed = new DeepLearning4jDistributed(iterator,stateTracker);
        distributed.setup(conf);

    }


    public void tearDown() throws Exception {
      if(stateTracker != null) {
          stateTracker.finish();
          stateTracker.shutdown();
          distributed.shutdown();
      }

        if(new File("model-saver").exists())
            new File("model-saver").delete();
        Thread.sleep(10000);
    }


    public  Configuration createConfiguration() {
        Configuration conf = new Configuration();
        conf.set(WorkerPerformerFactory.WORKER_PERFORMER, workPerformFactoryClassName);
        return conf;
    }

    public StateTracker createStateTracker() throws  Exception {
        if(stateTracker != null) {
            stateTracker.finish();
            stateTracker.shutdown();
        }
        return new HazelCastStateTracker();
    }


    /**
     * Name of the work performer class
     * @return
     */
    public abstract String workPerformFactoryClassName();


    /**
     *
     * @return
     */
    public abstract JobIterator createIterator();


}
