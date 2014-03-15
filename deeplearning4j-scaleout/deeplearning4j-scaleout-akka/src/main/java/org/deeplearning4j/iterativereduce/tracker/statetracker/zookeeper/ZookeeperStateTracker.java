package org.deeplearning4j.iterativereduce.tracker.statetracker.zookeeper;

import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.LinkedBlockingQueue;

import org.apache.curator.RetryPolicy;
import org.apache.curator.ensemble.EnsembleProvider;
import org.apache.curator.ensemble.fixed.FixedEnsembleProvider;
import org.apache.curator.framework.CuratorFramework;
import org.apache.curator.framework.CuratorFrameworkFactory;
import org.apache.curator.framework.recipes.queue.DistributedDelayQueue;
import org.apache.curator.framework.recipes.queue.DistributedIdQueue;
import org.apache.curator.framework.recipes.queue.QueueBuilder;
import org.apache.curator.framework.recipes.queue.QueueConsumer;
import org.apache.curator.framework.recipes.queue.QueueSerializer;
import org.apache.curator.framework.state.ConnectionState;
import org.apache.curator.retry.RetryUntilElapsed;
import org.apache.zookeeper.CreateMode;
import org.apache.zookeeper.data.Stat;
import org.deeplearning4j.iterativereduce.actor.core.Job;
import org.deeplearning4j.iterativereduce.actor.core.actor.WorkerState;
import org.deeplearning4j.iterativereduce.tracker.statetracker.StateTracker;
import org.deeplearning4j.scaleout.zookeeper.utils.ZkUtils;

@SuppressWarnings("unchecked")
public class ZookeeperStateTracker implements StateTracker, QueueConsumer<WorkerState>,QueueSerializer<WorkerState> {

	public final static String JOBS = "org.deeplearning4j.jobs";
	public final static String WORKERS = "org.deeplearning4j.workers";
	public final static String CURRENT_WORKERS = "WORKERS";
	public final static String AVAILABLE_WORKERS = "AVAILABLE_WORKERS";

	private LinkedBlockingQueue<WorkerState> availableWorkers = new LinkedBlockingQueue<>();

	public final static String CURRENT_JOBS = "JOBS";
	private CuratorFramework jobClient,workersClient;
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

		queue = QueueBuilder.builder(workersClient, this, this, availableWorkersPath()).buildIdQueue();

		queue.start();
	}

	@Override
	public void addJobToCurrent(Job j) throws Exception {
		Stat stat = jobClient.checkExists().forPath(currentJobsPath());


		if(stat != null && stat.getDataLength() != 0) { 
			byte[] data = jobClient.getZookeeperClient().getZooKeeper().getData( currentJobsPath() , false, stat);

			List<Job> jobs = ZkUtils.fromBytes(data, List.class);
			jobs.add(j);
			jobClient.delete().forPath(currentJobsPath());
			jobClient.create().creatingParentsIfNeeded().withMode(CreateMode.PERSISTENT)
			.forPath(currentJobsPath(), ZkUtils.toBytes((Serializable) jobs));


		}

		else {
			List<Job> currentJobs = new ArrayList<>();
			currentJobs.add(j);
			jobClient.create().creatingParentsIfNeeded().withMode(CreateMode.PERSISTENT).forPath(currentJobsPath(), ZkUtils.toBytes((Serializable) currentJobs));

		}

	}

	@Override
	public Map<String, WorkerState> currentWorkers() throws Exception {
		Stat stat = workersClient.checkExists().forPath(currentWorkersPath());
		if(stat != null && stat.getDataLength() != 0) { 
			byte[] data = workersClient.getData().forPath(currentWorkersPath());

			Map<String, WorkerState> workers = ZkUtils.fromBytes(data, Map.class);
			return workers;

		}

		return Collections.emptyMap();
	}

	@Override
	public WorkerState nextAvailableWorker() throws Exception {
		return availableWorkers.take();
	}


	private String currentJobsPath() {
		return "/" + CURRENT_JOBS;
	}

	private String currentWorkersPath() {
		return  "/" + CURRENT_WORKERS;
	}

	private String availableWorkersPath() {
		return CuratorFrameworkFactory.builder().namespace(WORKERS) + "/" + AVAILABLE_WORKERS;
	}

	@Override
	public void requeueJob(Job j) throws Exception {
		Stat stat = jobClient.checkExists().forPath(currentJobsPath());
		if(stat != null && stat.getDataLength() != 0) { 
			byte[] data = jobClient.getZookeeperClient().getZooKeeper().getData( currentJobsPath() , false, stat);

			List<Job> jobs = ZkUtils.fromBytes(data, List.class);
			jobs.add(j);
			jobClient.delete().forPath(currentJobsPath());
			jobClient.create().creatingParentsIfNeeded().withMode(CreateMode.PERSISTENT).forPath(currentJobsPath(), ZkUtils.toBytes((Serializable) jobs));


		}

		else {
			List<Job> currentJobs = new ArrayList<>();
			currentJobs.add(j);
			jobClient.create().creatingParentsIfNeeded().withMode(CreateMode.PERSISTENT).forPath(currentJobsPath(), ZkUtils.toBytes((Serializable) currentJobs));

		}


	}

	@Override
	public void setWorkerDone(String id) throws Exception {
		Stat stat = workersClient.checkExists().forPath(currentWorkersPath());
		if(stat != null && stat.getDataLength() != 0) { 
			Map<String, WorkerState> workers = currentWorkers();
			WorkerState state = workers.get(id);
			if(state != null) {
				state.setAvailable(true);
			}
			workersClient.delete().forPath(currentWorkersPath());
			workersClient.create().creatingParentsIfNeeded().withMode(CreateMode.PERSISTENT).forPath(currentWorkersPath(), ZkUtils.toBytes((Serializable) workers));
			queue.put(state, id);


		}


	}

	@Override
	public void clearWorker(WorkerState worker) throws Exception {
		Stat stat = workersClient.checkExists().forPath(currentWorkersPath());
		if(stat != null && stat.getDataLength() != 0) { 
			byte[] data = workersClient.getZookeeperClient().getZooKeeper().getData( currentJobsPath() , false, stat);
			Map<String, WorkerState> workers = ZkUtils.fromBytes(data, Map.class);
			workers.remove(worker.getWorkerId());
			workersClient.create().creatingParentsIfNeeded().withMode(CreateMode.PERSISTENT).forPath(currentWorkersPath(), ZkUtils.toBytes((Serializable) workers));
		}


	}

	@Override
	public void addWorker(WorkerState worker) throws Exception {
		Stat stat = workersClient.checkExists().forPath(currentWorkersPath());

		if(stat != null && stat.getDataLength() != 0) { 
			byte[] data = workersClient.getZookeeperClient().getZooKeeper().getData( currentJobsPath() , false, stat);
			Map<String, WorkerState> workers = ZkUtils.fromBytes(data, Map.class);
			workersClient.delete().forPath(currentWorkersPath());
			workersClient.create().creatingParentsIfNeeded().withMode(CreateMode.PERSISTENT).forPath(currentWorkersPath(), ZkUtils.toBytes((Serializable) workers));
			queue.put(worker, worker.getWorkerId());

		}

		else {

			Map<String, WorkerState> workers = new HashMap<String,WorkerState>();
			workers.put(worker.getWorkerId(),worker);
			workersClient.create().creatingParentsIfNeeded().withMode(CreateMode.PERSISTENT).forPath(currentWorkersPath(), ZkUtils.toBytes((Serializable) workers));
			queue.put(worker, worker.getWorkerId());


		}

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
		Stat stat = jobClient.checkExists().forPath(currentJobsPath());


		if(stat != null && stat.getDataLength() != 0) { 
			byte[] data = jobClient.getData().forPath(currentJobsPath());

			List<Job> jobs = ZkUtils.fromBytes(data, List.class);
			return jobs;

		}

		return Collections.emptyList();
	}



	@Override
	public void stateChanged(CuratorFramework arg0, ConnectionState arg1) {
		// TODO Auto-generated method stub

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
		List<Job> jobs = currentJobs();
		jobs.remove(j);
		jobClient.delete().forPath(currentJobsPath());
		jobClient.create().creatingParentsIfNeeded().withMode(CreateMode.PERSISTENT).forPath(currentJobsPath(), ZkUtils.toBytes((Serializable) jobs));

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




}
