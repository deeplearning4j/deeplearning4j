package org.deeplearning4j.iterativereduce.tracker.statetracker.zookeeper;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicReference;

import org.deeplearning4j.iterativereduce.actor.core.Job;
import org.deeplearning4j.iterativereduce.actor.core.actor.WorkerState;
import org.deeplearning4j.iterativereduce.tracker.statetracker.StateTracker;
import org.deeplearning4j.scaleout.iterativereduce.multi.UpdateableImpl;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@SuppressWarnings("unchecked")
public class ZookeeperStateTracker implements StateTracker<UpdateableImpl> {

	/**
	 * 
	 */
	private static final long serialVersionUID = -7374372180080957334L;
	public final static String JOBS = "org.deeplearning4j.jobs";
	public final static String WORKERS = "org.deeplearning4j.workers";
	public final static String CURRENT_WORKERS = "WORKERS";
	public final static String AVAILABLE_WORKERS = "AVAILABLE_WORKERS";
	public final static String TOPICS = "topics";
	public final static String RESULT = "RESULT";
	public final static String RESULT_LOC = "RESULT_LOC";
	private volatile AtomicReference<UpdateableImpl> master;
	private volatile LinkedBlockingQueue<Job> jobs = new LinkedBlockingQueue<>();
	private volatile Map<String,WorkerState> workers = new HashMap<String,WorkerState>();
	private volatile  List<String> topics =  new ArrayList<>();
	private volatile List<Job> redist = new ArrayList<>();
	private static Logger log = LoggerFactory.getLogger(ZookeeperStateTracker.class);


	private volatile LinkedBlockingQueue<WorkerState> availableWorkers = new LinkedBlockingQueue<>();

	public final static String CURRENT_JOBS = "JOBS";


	public ZookeeperStateTracker() throws Exception {
		this("localhost:2181");
	}

	public ZookeeperStateTracker(String connectionString) throws Exception {
		super();

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
		return availableWorkers.take();
	}




	@Override
	public void requeueJob(Job j) throws Exception {
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
		UpdateableImpl u =  master.get();
		return u.clone();
	}

	@Override
	public  void setCurrent(UpdateableImpl e) throws Exception {
		if(this.master == null)
			this.master = new AtomicReference<UpdateableImpl>(e);
		else
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
