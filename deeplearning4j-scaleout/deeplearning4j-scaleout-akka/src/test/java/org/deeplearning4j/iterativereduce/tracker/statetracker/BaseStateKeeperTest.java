package org.deeplearning4j.iterativereduce.tracker.statetracker;

import static org.junit.Assert.*;

import java.util.List;
import java.util.Map;
import java.util.UUID;

import org.deeplearning4j.iterativereduce.actor.core.Job;
import org.deeplearning4j.iterativereduce.actor.core.actor.WorkerState;
import org.junit.Before;
import org.junit.Test;

public abstract class BaseStateKeeperTest {

	protected StateTracker tracker;


	
	public void init() {
		tracker = createTracker();
	}



	@Test
	public void testWorkerRetrieval() throws Exception {

		String id = UUID.randomUUID().toString();
		WorkerState state = new WorkerState(id, null);
		state.setAvailable(true);
		tracker.addWorker(state);

		Map<String, WorkerState> workers = tracker.currentWorkers();

		assertEquals(1,workers.size());
	}


	@Test
	public void ensureJobRetrieval() throws Exception {

		String id = UUID.randomUUID().toString();
		Job j = new Job(false, id, id, false);
		tracker.addJobToCurrent(j);

		List<Job> jobs = tracker.currentJobs();
		assertEquals(1,jobs.size());

		tracker.clearJob(j);

		jobs = tracker.currentJobs();

		assertEquals(true,jobs.isEmpty());


	}


	public StateTracker getTracker() {
		return tracker;
	}

	public abstract StateTracker createTracker();

}
