package org.deeplearning4j.iterativereduce.tracker.statetracker.hazelcast;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Queue;

import org.deeplearning4j.iterativereduce.actor.core.Job;
import org.deeplearning4j.iterativereduce.actor.core.actor.WorkerState;
import org.deeplearning4j.iterativereduce.tracker.statetracker.StateTracker;
import org.deeplearning4j.scaleout.iterativereduce.multi.UpdateableImpl;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.hazelcast.client.HazelcastClient;
import com.hazelcast.client.config.ClientConfig;
import com.hazelcast.config.Config;
import com.hazelcast.config.GroupConfig;
import com.hazelcast.config.TcpIpConfig;
import com.hazelcast.core.Hazelcast;
import com.hazelcast.core.HazelcastInstance;
import com.hazelcast.core.IAtomicReference;

public class HazelCastStateTracker implements StateTracker<UpdateableImpl> {

	/**
	 * 
	 */
	private static final long serialVersionUID = -7374372180080957334L;
	public final static String JOBS = "org.deeplearning4j.jobs";
	public final static String REDIST = "redist";
	public final static String WORKERS = "org.deeplearning4j.workers";
	public final static String CURRENT_WORKERS = "WORKERS";
	public final static String AVAILABLE_WORKERS = "AVAILABLE_WORKERS";
	public final static String TOPICS = "topics";
	public final static String RESULT = "RESULT";
	public final static String RESULT_LOC = "RESULT_LOC";
	private volatile IAtomicReference<Object> master;
	private volatile List<Job> jobs;
	private volatile Map<String,WorkerState> workers;
	private volatile  List<String> topics;
	private volatile List<Job> redist;
	private static Logger log = LoggerFactory.getLogger(HazelCastStateTracker.class);
	private Config config;
	private volatile Queue<WorkerState> availableWorkers;
	public final static int DEFAULT_HAZELCAST_PORT = 2510;
	public final static String CURRENT_JOBS = "JOBS";
	private HazelcastInstance h;

	public HazelCastStateTracker() throws Exception {
		this("localhost:2181");

	}

	public HazelCastStateTracker(String connectionString) throws Exception {
		String[] s = connectionString.split(":");
		if(!s.equals("localhost") && !s.equals("127.0.0.1") && !s.equals("0.0.0.0")) {
			ClientConfig clientConfig = new ClientConfig();
			clientConfig.getNetworkConfig().setAddresses(Arrays.asList(s[0] + ":" + DEFAULT_HAZELCAST_PORT));
			h = HazelcastClient.newHazelcastClient(clientConfig);

		}
		else {
			config = new Config();
			h = Hazelcast.newHazelcastInstance(config);

		}

		jobs = h.getList(JOBS);
		workers = h.getMap(CURRENT_WORKERS);
		topics = h.getList(TOPICS);
		redist = h.getList(REDIST);
		availableWorkers = h.getQueue(AVAILABLE_WORKERS);
		master = h.getAtomicReference(RESULT);
		h.getAtomicReference(RESULT);


	}

	@Override
	public void addJobToCurrent(Job j) throws Exception {
		jobs.add(j);
		workers.get(j.getWorkerId()).setAvailable(false);

	}

	@Override
	public Map<String, WorkerState> currentWorkers() throws Exception {
		return workers;

	}



	@Override
	public  WorkerState nextAvailableWorker() throws Exception {
		return availableWorkers.poll();
	}




	@Override
	public void requeueJob(Job j) throws Exception {
		jobs.remove(j);
		redist.add(j);
	}

	@Override
	public void setWorkerDone(String id) throws Exception {
		this.workers.get(id).setAvailable(true);
		availableWorkers.add(workers.get(id));


	}

	@Override
	public void clearWorker(WorkerState worker) throws Exception {
		this.workers.remove(worker.getWorkerId());
		availableWorkers.remove(worker);
	}

	@Override
	public void addWorker(WorkerState worker) throws Exception {
		workers.put(worker.getWorkerId(),worker);
		availableWorkers.add(worker);


	}









	@Override
	public boolean everyWorkerAvailable() throws Exception {
		Map<String, WorkerState> workers = currentWorkers();
		for(WorkerState state : workers.values())
			if(!state.isAvailable())
				return false;
		return true;
	}

	@Override
	public List<Job> currentJobs() throws Exception {
		return new ArrayList<Job>(jobs);
	}




	@Override
	public void clearJob(Job j) throws Exception {
		jobs.remove(j);
	}

	@Override
	public void shutdown() {


	}

	@Override
	public void addTopic(String topic) throws Exception {
		topics.add(topic);


	}

	@Override
	public List<String> topics() throws Exception {
		return topics;
	}

	@Override
	public  UpdateableImpl getCurrent() throws Exception {
		UpdateableImpl u =  (UpdateableImpl) master.get();
		return u.clone();
	}

	@Override
	public  void setCurrent(UpdateableImpl e) throws Exception {
		this.master.set(e);
	}

	@Override
	public List<Job> jobsToRedistribute() {
		return redist;
	}

	@Override
	public void jobRequeued(Job j) {
		redist.remove(j);
	}




}
