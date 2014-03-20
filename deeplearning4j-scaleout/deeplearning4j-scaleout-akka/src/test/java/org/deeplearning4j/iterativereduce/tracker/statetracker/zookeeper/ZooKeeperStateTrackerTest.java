package org.deeplearning4j.iterativereduce.tracker.statetracker.zookeeper;

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

public class ZooKeeperStateTrackerTest  {

	protected StateTracker tracker;

	@Before
	public void init() {
		

	}





	@Test
	public void testClientServer() throws Exception {
		HazelCastStateTracker master = new HazelCastStateTracker("localhost:" + HazelCastStateTracker.DEFAULT_HAZELCAST_PORT,"master");

		master.addWorker("id");
		master.moveToFinetune();
		
		
	
		
		HazelCastStateTracker worker = new HazelCastStateTracker("localhost:" + HazelCastStateTracker.DEFAULT_HAZELCAST_PORT,"worker");

		
		assertEquals(master.isPretrain(),worker.isPretrain());
		assertEquals(master.numWorkers(),worker.numWorkers());
		
		
	}




}
