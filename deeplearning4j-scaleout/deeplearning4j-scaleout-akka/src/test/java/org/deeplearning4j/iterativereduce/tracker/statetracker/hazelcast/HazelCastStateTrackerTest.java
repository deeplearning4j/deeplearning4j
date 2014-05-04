package org.deeplearning4j.iterativereduce.tracker.statetracker.hazelcast;

import static org.junit.Assert.assertEquals;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.UUID;

import org.apache.curator.test.TestingServer;
import org.deeplearning4j.iterativereduce.actor.core.Job;
import org.deeplearning4j.iterativereduce.actor.core.actor.WorkerState;
import org.deeplearning4j.iterativereduce.tracker.statetracker.StateTracker;
import org.deeplearning4j.iterativereduce.tracker.statetracker.hazelcast.HazelCastStateTracker;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import com.hazelcast.core.Hazelcast;

public class HazelCastStateTrackerTest  {




    @Test
    public void testReplication() throws Exception {
        HazelCastStateTracker master = new HazelCastStateTracker("localhost:" + HazelCastStateTracker.DEFAULT_HAZELCAST_PORT,"master",HazelCastStateTracker.DEFAULT_HAZELCAST_PORT);
        master.addReplicate("1");
        assertEquals(true,master.needsReplicate("1"));
        master.doneReplicating("1");
        assertEquals(false,master.needsReplicate("1"));
        master.shutdown();
    }
	
	@Test
    public void testJob() throws Exception {
        HazelCastStateTracker master = new HazelCastStateTracker("localhost:" + HazelCastStateTracker.DEFAULT_HAZELCAST_PORT,"master",HazelCastStateTracker.DEFAULT_HAZELCAST_PORT);
        Job j = new Job("1","hi");
        master.addJobToCurrent(j);
        assertEquals(true,master.jobFor("1") != null);
        master.clearJob(j.getWorkerId());
        assertEquals(true,master.jobFor("1") == null);
        master.shutdown();
    }


	@Test
	public void testClientServer() throws Exception {
		HazelCastStateTracker master = new HazelCastStateTracker("localhost:" + HazelCastStateTracker.DEFAULT_HAZELCAST_PORT,"master",HazelCastStateTracker.DEFAULT_HAZELCAST_PORT);
        assertEquals(true,master.isPretrain());
        assertEquals(false,master.isDone());
		master.runPreTrainIterations(10);
		master.addWorker("id");
		master.moveToFinetune();
		
		
	
		
		HazelCastStateTracker worker = new HazelCastStateTracker("localhost:" + HazelCastStateTracker.DEFAULT_HAZELCAST_PORT,"worker",HazelCastStateTracker.DEFAULT_HAZELCAST_PORT);

		assertEquals(master.isPretrain(),worker.isPretrain());
		assertEquals(master.numWorkers(),worker.numWorkers());
		assertEquals(10,worker.runPreTrainIterations());

        master.shutdown();
        worker.shutdown();

		
	}




}
