package org.deeplearning4j.iterativereduce.tracker.statetracker.zookeeper;

import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;

import org.apache.curator.ensemble.fixed.FixedEnsembleProvider;
import org.apache.curator.framework.CuratorFramework;
import org.apache.curator.framework.CuratorFrameworkFactory;
import org.apache.curator.framework.recipes.queue.DistributedIdQueue;
import org.apache.curator.framework.recipes.queue.QueueBuilder;
import org.apache.curator.framework.recipes.queue.QueueConsumer;
import org.apache.curator.framework.recipes.queue.QueueSerializer;
import org.apache.curator.framework.state.ConnectionState;
import org.apache.curator.retry.RetryUntilElapsed;
import org.apache.curator.utils.EnsurePath;
import org.apache.zookeeper.CreateMode;
import org.apache.zookeeper.data.Stat;
import org.deeplearning4j.iterativereduce.actor.core.Job;
import org.deeplearning4j.iterativereduce.actor.core.actor.WorkerState;
import org.deeplearning4j.iterativereduce.tracker.statetracker.StateTracker;
import org.deeplearning4j.scaleout.iterativereduce.Updateable;
import org.deeplearning4j.scaleout.iterativereduce.multi.UpdateableImpl;
import org.deeplearning4j.scaleout.zookeeper.utils.ZkUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@SuppressWarnings("unchecked")
public class ZookeeperStateTracker implements StateTracker<Updateable<?>>, QueueConsumer<WorkerState>,QueueSerializer<WorkerState> {

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
	private Updateable<?> master;
	private static Logger log = LoggerFactory.getLogger(ZookeeperStateTracker.class);


	private LinkedBlockingQueue<WorkerState> availableWorkers = new LinkedBlockingQueue<>();

	public final static String CURRENT_JOBS = "JOBS";
	private CuratorFramework jobClient,workersClient,topicsClient,resultClient;
	private DistributedIdQueue<WorkerState>   queue;


	public ZookeeperStateTracker() throws Exception {
		this("localhost:2181");
	}

	public ZookeeperStateTracker(String connectionString) throws Exception {
		super();
		jobClient = CuratorFrameworkFactory.builder().namespace(JOBS)
				.retryPolicy(new RetryUntilElapsed(60000, 5000))
				.ensembleProvider(new FixedEnsembleProvider(connectionString)).build();
		jobClient.start();

		jobClient.getZookeeperClient().blockUntilConnectedOrTimedOut();


		workersClient = CuratorFrameworkFactory.builder().namespace(WORKERS)
				.retryPolicy(new RetryUntilElapsed(60000, 5000))
				.ensembleProvider(new FixedEnsembleProvider(connectionString))
				.build();

		workersClient.start();
		workersClient.getZookeeperClient().blockUntilConnectedOrTimedOut();


		topicsClient = CuratorFrameworkFactory.builder().namespace(TOPICS)
				.retryPolicy(new RetryUntilElapsed(60000, 5000))
				.ensembleProvider(new FixedEnsembleProvider(connectionString))
				.build();


		topicsClient.start();
		topicsClient.getZookeeperClient().blockUntilConnectedOrTimedOut();


		resultClient = CuratorFrameworkFactory.builder().namespace(RESULT)
				.retryPolicy(new RetryUntilElapsed(60000, 5000)).connectionTimeoutMs(99999999)
				.ensembleProvider(new FixedEnsembleProvider(connectionString))
				.build();

		resultClient.start();
		resultClient.getZookeeperClient().blockUntilConnectedOrTimedOut();






		queue = QueueBuilder.builder(workersClient, this, this, availableWorkersPath()).buildIdQueue();

		queue.start();
	}

	@Override
	public void addJobToCurrent(Job j) throws Exception {
		if(j.getWork() == null) {
			log.info("Ignoring null work");
			return;
		}


		List<Job> jobs = currentJobs();
		jobs.add(j);
		createJobsWithData(jobs);


	}

	@Override
	public Map<String, WorkerState> currentWorkers() throws Exception {
		return currentWorkers(0);
	}

	private Map<String,WorkerState> currentWorkers(int numRetries) throws Exception {
		Stat stat = workersClient.checkExists().forPath(currentWorkersPath());
		if(numRetries >= 3) {
			return new HashMap<>();
		}

		numRetries++;
		if(stat != null && stat.getDataLength() != 0) { 
			byte[] data = workersClient.getData().forPath(currentWorkersPath());

			try {
				Map<String, WorkerState> workers = ZkUtils.fromBytes(data, Map.class);
				return workers;

			}catch(Exception e) {
				return currentWorkers(numRetries);
			}


		}

		return new HashMap<>();
	}


	@Override
	public synchronized WorkerState nextAvailableWorker() throws Exception {
		WorkerState ret =  availableWorkers.poll(30, TimeUnit.SECONDS);
		while(ret == null) {
			Map<String, WorkerState> workers = currentWorkers();
			for(WorkerState w : workers.values()) {
				if(w.isAvailable())
					availableWorkers.add(w);
			}

			ret =  availableWorkers.poll(30, TimeUnit.SECONDS);

		}

		return ret;
	}


	private String resultsPath() {
		return "/" + RESULT_LOC;
	}

	private String topicsPath() {
		return "/" + TOPICS;
	}

	private String currentJobsPath() {
		return "/" + CURRENT_JOBS;
	}

	private String currentWorkersPath() {
		return  "/" + CURRENT_WORKERS;
	}

	private String availableWorkersPath() {
		return  "/" + AVAILABLE_WORKERS;
	}

	@Override
	public void requeueJob(Job j) throws Exception {
		if(j.getWork() == null) {
			log.info("Ignoring null work");
			return;
		}

		List<Job> currentJobs = new ArrayList<>();
		currentJobs.add(j);
		createJobsWithData(currentJobs);





	}

	@Override
	public void setWorkerDone(String id) throws Exception {
		Stat stat = workersClient.checkExists().forPath(currentWorkersPath());
		if(stat != null && stat.getDataLength() != 0) { 
			Map<String, WorkerState> workers = currentWorkers();
			WorkerState state = workers.get(id);
			if(state != null) {
				state.setAvailable(true);
				createWorkersWithData(workers);
				queue.put(state, id);

			}


		}


	}

	@Override
	public void clearWorker(WorkerState worker) throws Exception {
		Stat stat = workersClient.checkExists().forPath(currentWorkersPath());
		if(stat != null && stat.getDataLength() != 0) { 
			byte[] data = workersClient.getData().forPath( currentJobsPath());
			Map<String, WorkerState> workers = ZkUtils.fromBytes(data, Map.class);
			workers.remove(worker.getWorkerId());
			createWorkersWithData(workers);
		}


	}

	@Override
	public void addWorker(WorkerState worker) throws Exception {
		Map<String, WorkerState> workers = this.currentWorkers();
		createWorkersWithData(workers);
		queue.put(worker, worker.getWorkerId());


	}



	private void createJobsWithData(List<Job> jobs) {
		createJobsWithData(jobs,0);
	}




	private void createJobsWithData(List<Job> currentJobs,int numTimes) {
		if(numTimes >= 3) {
			log.warn("Exiting...>= 3 attempts at adding data in to current jobs");
			return;
		}

		numTimes++;

		try {

			EnsurePath ensurePath = new EnsurePath(currentJobsPath());
			ensurePath.ensure(jobClient.getZookeeperClient());
			if(jobClient.checkExists().forPath(currentJobsPath()) != null) {
				jobClient.setData().forPath(currentJobsPath(), ZkUtils.toBytes((Serializable) currentJobs));
			}

			else
				jobClient.create().forPath(currentJobsPath(),ZkUtils.toBytes((Serializable) currentJobs));

		}catch(Exception e) {
			createJobsWithData(currentJobs,numTimes);
		}

	}

	private void createWorkersWithData(Map<String,WorkerState> workers,int numTimes) {
		if(numTimes >= 3) {
			log.warn("3 failures to insert workers with zk creation...exiting");
			return;
		}

		numTimes++;


		try {

			EnsurePath ensurePath = new EnsurePath(currentWorkersPath());
			ensurePath.ensure(workersClient.getZookeeperClient());

			if(workersClient.checkExists().forPath(currentWorkersPath()) != null)
				workersClient.setData().forPath(currentWorkersPath(),ZkUtils.toBytes((Serializable) workers));

			else
				workersClient.create().creatingParentsIfNeeded().withMode(CreateMode.PERSISTENT).forPath(currentWorkersPath(), ZkUtils.toBytes((Serializable) workers));

		}catch(Exception e) {
			createWorkersWithData(workers,numTimes);
		}
	}

	private void createWorkersWithData(Map<String,WorkerState> workers) {
		createWorkersWithData(workers,0);
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
		return currentJobs(0);
	}


	public List<Job> currentJobs(int numRetries) throws Exception {
		if(numRetries >= 3) {
			log.warn("Unable to obtain current jobs, exitig");
			return Collections.emptyList();
		}

		if(jobClient.checkExists().forPath(currentJobsPath()) != null) { 
			byte[] data = jobClient.getData().forPath(currentJobsPath());
			try {
				List<Job> jobs = ZkUtils.fromBytes(data, List.class);
				return jobs;


			}catch(Exception e) {
				return currentJobs(numRetries);
			}

		}

		return new ArrayList<>();
	}


	@Override
	public void stateChanged(CuratorFramework arg0, ConnectionState arg1) {

	}

	@Override
	public void consumeMessage(WorkerState arg0) throws Exception {
		availableWorkers.add(arg0);			
	}

	@Override
	public WorkerState deserialize(byte[] arg0) {
		try {
			return ZkUtils.fromBytes(arg0, WorkerState.class);
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}

	@Override
	public byte[] serialize(WorkerState arg0) {
		try {
			return ZkUtils.toBytes(arg0);
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}

	@Override
	public void clearJob(Job j) throws Exception {
		if(j.getWork() == null) {
			log.info("Ignoring null work");
			return;
		}

		List<Job> jobs = currentJobs();
		jobs.remove(j);
		createJobsWithData(jobs);

	}

	@Override
	public void shutdown() {
		if(jobClient != null)
			jobClient.close();
		if(workersClient != null)
			workersClient.close();
		if(queue != null)
			try {
				queue.close();
			} catch (IOException e) {
				throw new RuntimeException(e);
			}

	}

	@Override
	public void addTopic(String topic) throws Exception {
		List<String> topics = topics();
		topics.add(topic);
		if(topicsClient.checkExists().forPath(topicsPath()) != null) {
			topicsClient.setData().forPath(topicsPath(), ZkUtils.toBytes((Serializable) topics));
		}

		else {
			topicsClient.create().creatingParentsIfNeeded().forPath(topicsPath(),ZkUtils.toBytes((Serializable) topics));

		}



	}

	@Override
	public List<String> topics() throws Exception {
		if(topicsClient.checkExists().forPath(topicsPath()) != null) {
			byte[] data = topicsClient.getData().forPath(topicsPath());
			List<String> topics = ZkUtils.fromBytes(data, List.class);
			return topics;
		}

		return Collections.emptyList();
	}

	@Override
	public synchronized Updateable<?> getCurrent() throws Exception {
		UpdateableImpl u = (UpdateableImpl) master;
		return u.clone();
	}

	@Override
	public synchronized void setCurrent(Updateable<?> e) throws Exception {
		this.master = e;

	}




}
