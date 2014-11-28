package org.deeplearning4j.scaleout.actor;

import static org.junit.Assert.*;

import akka.actor.ActorRef;
import akka.actor.ActorSystem;
import akka.actor.Props;
import org.deeplearning4j.scaleout.actor.core.actor.WorkerActor;
import org.deeplearning4j.scaleout.conf.Configuration;
import org.deeplearning4j.scaleout.conf.DeepLearningConfigurable;
import org.deeplearning4j.scaleout.job.Job;
import org.deeplearning4j.scaleout.statetracker.StateTracker;
import org.deeplearning4j.scaleout.statetracker.hazelcast.HazelCastStateTracker;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

/**
 * Created by agibsonccc on 11/27/14.
 */
public class WorkerActorTest implements DeepLearningConfigurable {

    private StateTracker stateTracker;
    private ActorSystem testSystem;
    private Props test;
    private Configuration conf;
    private TestPerformer testPerformer;

    @Before
    public void before() throws Exception {
        stateTracker = new HazelCastStateTracker();
        testSystem = ActorSystem.create();
        conf = new Configuration();
        testPerformer = new TestPerformer();
        test = WorkerActor.propsFor(conf, stateTracker,testPerformer);


    }

    @After
    public void after() {
        stateTracker.finish();
        testSystem.shutdown();
    }


    @Test
    public void testActor() throws Exception {
        testSystem.actorOf(test);
        Thread.sleep(10000);
        assertTrue(!stateTracker.workers().isEmpty());
        String id = stateTracker.workers().get(0);
        Job testJob = new Job("hello",id);
        stateTracker.addJobToCurrent(testJob);
        while(!testPerformer.isPerformCalled()) {
            Thread.sleep(10000);
        }



    }


    @Override
    public void setup(Configuration conf) {

    }
}
