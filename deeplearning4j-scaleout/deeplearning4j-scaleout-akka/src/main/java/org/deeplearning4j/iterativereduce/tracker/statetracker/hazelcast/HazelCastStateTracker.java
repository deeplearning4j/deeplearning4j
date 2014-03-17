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

import akka.cluster.protobuf.msg.ClusterMessages.Join;

import com.hazelcast.client.HazelcastClient;
import com.hazelcast.client.config.ClientConfig;
import com.hazelcast.config.Config;
import com.hazelcast.config.JoinConfig;
import com.hazelcast.config.ListConfig;
import com.hazelcast.config.MapConfig;
import com.hazelcast.config.NetworkConfig;
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
	public final static String IS_PRETRAIN = "ispretrain";
	public final static String RESULT_LOC = "RESULT_LOC";
	private volatile IAtomicReference<Object> master;
	private volatile List<Job> jobs;
	private volatile Map<String,WorkerState> workers;
	private volatile  List<String> topics;
	private volatile List<Job> redist;
	private volatile IAtomicReference<Object> isPretrain;
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
		if(!s[0].equals("localhost") && !s[0].equals("127.0.0.1") && !s[0].equals("0.0.0.0")) {
			ClientConfig clientConfig = new ClientConfig();
			clientConfig.getNetworkConfig().setAddresses(Arrays.asList(s[0] + ":" + DEFAULT_HAZELCAST_PORT));
			h = HazelcastClient.newHazelcastClient(clientConfig);

		}
		else {
			config =  hazelcast();			
			h = Hazelcast.newHazelcastInstance(config);

		}



		jobs = h.getList(JOBS);
		workers = h.getMap(CURRENT_WORKERS);
		topics = h.getList(TOPICS);
		redist = h.getList(REDIST);
		availableWorkers = h.getQueue(AVAILABLE_WORKERS);
		master = h.getAtomicReference(RESULT);
		isPretrain = h.getAtomicReference(IS_PRETRAIN);
		isPretrain.set(false);

	}

	private Config hazelcast() {
		Config conf = new Config();
		conf.getNetworkConfig().setPort(DEFAULT_HAZELCAST_PORT);
		conf.getNetworkConfig().setPortAutoIncrement(true);



		conf.setProperty("hazelcast.initial.min.cluster.size","1");


		JoinConfig join = conf.getNetworkConfig().getJoin();
		join.getTcpIpConfig().setEnabled(true);
		join.getAwsConfig().setEnabled(false);
		join.getMulticastConfig().setEnabled(false);


		join.getTcpIpConfig().setConnectionTimeoutSeconds(2000);
		join.getTcpIpConfig().addMember("127.0.0.1:5172");


		ListConfig jobConfig = new ListConfig();
		jobConfig.setName(JOBS);
		conf.addListConfig(jobConfig);

		MapConfig workersConfig = new MapConfig();
		workersConfig.setName(CURRENT_WORKERS);

		conf.addMapConfig(workersConfig);


		ListConfig topicsConfig = new ListConfig();
		topicsConfig.setName(TOPICS);

		conf.addListConfig(topicsConfig);

		ListConfig reDistConfig = new ListConfig();
		reDistConfig.setName(REDIST);

		conf.addListConfig(reDistConfig);


		ListConfig availableWorkersConfig = new ListConfig();
		availableWorkersConfig.setName(AVAILABLE_WORKERS);

		conf.addListConfig(availableWorkersConfig);



		return conf;

	}

	
	
	
	@Override
	public void addJobToCurrent(Job j) throws Exception {
		jobs.add(j);
		WorkerState w = workers.get(j.getWorkerId());
		w.setAvailable(false);
		workers.put(w.getWorkerId(), w);
	}

	@Override
	public Map<String, WorkerState> currentWorkers() throws Exception {
		return workers;

	}



	@Override
	public  WorkerState nextAvailableWorker() throws Exception {
		WorkerState ret = null;
		int timesLooped = 0;
		do {
			ret = availableWorkers.poll();
			timesLooped++;
			if(timesLooped % 5 == 0) {
				for(WorkerState w : workers.values()) {
					if(w.isAvailable() && !availableWorkers.contains(w))  {
						availableWorkers.add(w);
						log.info("Adding missing worker " + w.getWorkerId());
					}
				}
			}

		}while(ret == null);

		return ret;

	}




	@Override
	public void requeueJob(Job j) throws Exception {
		jobs.remove(j);
		redist.add(j);
	}

	@Override
	public void setWorkerDone(String id) throws Exception {
		WorkerState w = workers.get(id);
		w.setAvailable(true);
		workers.put(w.getWorkerId(), w);
		availableWorkers.add(workers.get(id));


	}

	@Override
	public void clearWorker(WorkerState worker) throws Exception {
		workers.remove(worker.getWorkerId());
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
		redist.remove(j);
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

	@Override
	public void jobDone(Job job) {
		jobs.remove(job);
		redist.remove(job);
	}

	@Override
	public boolean workerAvailable(String id) {
		WorkerState w = this.workers.get(id);
		if(w != null)
			return w.isAvailable();
		return false;
	}

	@Override
	public boolean isPretrain() {
		return (boolean) isPretrain.get();
	}

	@Override
	public void moveToFinetune() {
		isPretrain.set(false);
	}




}
