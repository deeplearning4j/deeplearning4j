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

public class ZooKeeperStateTrackerTest  {

	protected StateTracker tracker;

	@Before
	public void init() {
		try {
		
				tracker = createTracker();

		} catch (Exception e) {
			throw new RuntimeException(e);
		}

	}



	public StateTracker createTracker() {
		try {

			return new HazelCastStateTracker("localhost:2182","master");

		} catch (Exception e) {
			throw new RuntimeException(e);
		}

	}

	@After
	public void after() throws IOException {
		
		if(tracker != null)
			tracker.shutdown();
	}








	@Test
	public void testWorker() {
		StateTracker tracker = createTracker();
		tracker.addWorker("id");
		assertEquals(1,tracker.numWorkers());
		tracker.shutdown();
	}
	

	@Test
	public void ensureJobRetrieval() throws Exception {

		StateTracker tracker = createTracker();

		String id = UUID.randomUUID().toString();
		Job j = new Job(false, id, id, false);
		tracker.addJobToCurrent(j);

		List<Job> jobs = tracker.currentJobs();
		assertEquals(1,jobs.size());

		tracker.clearJob(j);

		jobs = tracker.currentJobs();

		assertEquals(true,jobs.isEmpty());
		
		tracker.shutdown();

	}
	
	
	


}
