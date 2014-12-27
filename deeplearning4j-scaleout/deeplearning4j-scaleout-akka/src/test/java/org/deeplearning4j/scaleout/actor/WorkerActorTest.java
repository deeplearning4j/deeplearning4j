package org.deeplearning4j.scaleout.actor;

import static org.junit.Assert.*;

import akka.actor.ActorSystem;
import akka.actor.Props;
import com.hazelcast.core.Hazelcast;
import org.deeplearning4j.scaleout.actor.core.actor.WorkerActor;
import org.deeplearning4j.scaleout.conf.Configuration;
import org.deeplearning4j.scaleout.conf.DeepLearningConfigurable;
import org.deeplearning4j.scaleout.job.Job;
import org.deeplearning4j.scaleout.api.statetracker.StateTracker;
import org.deeplearning4j.scaleout.statetracker.hazelcast.HazelCastStateTracker;
import org.junit.*;

/**
 * Created by agibsonccc on 11/27/14.
 */
public class WorkerActorTest implements DeepLearningConfigurable {

    private static StateTracker stateTracker;
    private static ActorSystem testSystem;
    private static Props test;
    private static Configuration conf;
    private static TestPerformer testPerformer;

    @BeforeClass
    public static void before() throws Exception {
        stateTracker = new HazelCastStateTracker();
        if(testSystem == null)
            testSystem = ActorSystem.create();
        conf = new Configuration();
        testPerformer = new TestPerformer();
        test = WorkerActor.propsFor(conf, stateTracker,testPerformer);


    }

    @AfterClass
    public static void after() {
        if(stateTracker != null) {
            stateTracker.finish();
            stateTracker.shutdown();
        }
        if(testSystem != null)
            testSystem.shutdown();
        Hazelcast.shutdownAll();
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
