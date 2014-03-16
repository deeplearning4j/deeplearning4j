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
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

public class ZooKeeperStateTrackerTest  {

	private TestingServer server;
	protected StateTracker tracker;

	@Before
	public void init() {
		try {
			if(server != null)
				server = new TestingServer(2182);
			if(tracker != null)
				tracker = createTracker();

		} catch (Exception e) {
			throw new RuntimeException(e);
		}

	}



	public StateTracker createTracker() {
		try {

			return new ZookeeperStateTracker("localhost:2182");

		} catch (Exception e) {
			throw new RuntimeException(e);
		}

	}

	@After
	public void after() throws IOException {
		if(server != null)
			server.stop();

		if(tracker != null)
			tracker.shutdown();
	}








	@Test
	public void testWorkerRetrieval() throws Exception {
		server = new TestingServer(2182);

		StateTracker tracker = createTracker();
		String id = UUID.randomUUID().toString();
		WorkerState state = new WorkerState(id);
		state.setAvailable(true);
		tracker.addWorker(state);

		Map<String, WorkerState> workers = tracker.currentWorkers();

		assertEquals(1,workers.size());
		WorkerState state2 = tracker.nextAvailableWorker();
		assertEquals(state,state2);
		
		tracker.shutdown();
		server.stop();
	}


	@Test
	public void ensureJobRetrieval() throws Exception {
		server = new TestingServer(2182);

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
		server.stop();

	}
	
	
	


}
