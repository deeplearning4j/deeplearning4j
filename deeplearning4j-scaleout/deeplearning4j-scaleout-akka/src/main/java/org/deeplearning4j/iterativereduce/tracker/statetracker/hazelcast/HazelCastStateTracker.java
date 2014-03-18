package org.deeplearning4j.iterativereduce.tracker.statetracker.hazelcast;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.deeplearning4j.iterativereduce.actor.core.Job;
import org.deeplearning4j.iterativereduce.actor.core.actor.WorkerState;
import org.deeplearning4j.iterativereduce.tracker.statetracker.StateTracker;
import org.deeplearning4j.scaleout.iterativereduce.multi.UpdateableImpl;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.hazelcast.client.HazelcastClient;
import com.hazelcast.client.config.ClientConfig;
import com.hazelcast.config.Config;
import com.hazelcast.config.JoinConfig;
import com.hazelcast.config.ListConfig;
import com.hazelcast.config.MapConfig;
import com.hazelcast.core.Hazelcast;
import com.hazelcast.core.HazelcastInstance;
import com.hazelcast.core.IAtomicReference;
import com.hazelcast.core.IList;
import com.hazelcast.core.ILock;
import com.hazelcast.core.IMap;
import com.hazelcast.core.TransactionalList;
import com.hazelcast.core.TransactionalMap;
import com.hazelcast.transaction.TransactionContext;

public class HazelCastStateTracker implements StateTracker<UpdateableImpl> {

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
	public final static String LOCKS = "LOCKS";
	public final static String IS_PRETRAIN = "ispretrain";
	public final static String RESULT_LOC = "RESULT_LOC";
	private volatile IAtomicReference<Object> master;
	private volatile IList<Job> jobs;
	private volatile IMap<String,WorkerState> workers;
	private volatile  IList<String> topics;
	private volatile IList<WorkerState> availableWorkers;
	private volatile IAtomicReference<Object> isPretrain;
	private static Logger log = LoggerFactory.getLogger(HazelCastStateTracker.class);
	private Config config;
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
		availableWorkers = h.getList(AVAILABLE_WORKERS);
		master = h.getAtomicReference(RESULT);
		isPretrain = h.getAtomicReference(IS_PRETRAIN);
		isPretrain.set(true);
		
		
	}

	private Config hazelcast() {
		Config conf = new Config();
		conf.getNetworkConfig().setPort(DEFAULT_HAZELCAST_PORT);
		conf.getNetworkConfig().setPortAutoIncrement(true);



		conf.setProperty("hazelcast.initial.min.cluster.size","1");
		conf.setProperty("hazelcast.shutdownhook.enabled","false");

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

	


		ListConfig availableWorkersConfig = new ListConfig();
		availableWorkersConfig.setName(AVAILABLE_WORKERS);
		conf.addListConfig(availableWorkersConfig);



		return conf;

	}

	private  TransactionContext beginTransaction() {
		TransactionContext context = h.newTransactionContext();


		context.beginTransaction();

		return context;
	}


	@Override
	public void addJobToCurrent(Job j) throws Exception {
		TransactionContext context = beginTransaction();
		context.getList(JOBS).add(j);
		IAtomicReference<Job> r = h.getAtomicReference("job-" + j.getWorkerId());
		while(r.get() != null) {
			log.info("Currently locked unable to add job for current worker");
		}
		
		r.set(j);
		context.commitTransaction();


	}

	@Override
	public Map<String, WorkerState> currentWorkers() throws Exception {
		return new HashMap<>(workers);

	}



	@Override
	public  WorkerState nextAvailableWorker() throws Exception {
		WorkerState ret = null;

		while(ret == null) {
			for(String key : workers.keySet()) {
				IAtomicReference<WorkerState> state = h.getAtomicReference(key);
				if(state.get().isAvailable()) {
					ret = workers.get(key);
					break;
				}
			}
		}

		return ret;

	}





	@Override
	public void clearWorker(WorkerState worker) throws Exception {
		TransactionContext context = beginTransaction();



		TransactionalMap<String,WorkerState> workers = context.getMap(CURRENT_WORKERS);


		workers.remove(worker.getWorkerId());

		
		
		
		h.getLock(worker.getWorkerId()).destroy();
		
		
		context.commitTransaction();
	}

	@Override
	public void addWorker(WorkerState worker) throws Exception {

		
		log.info("Added worker " + worker.getWorkerId());
		
		TransactionContext context = beginTransaction();

		TransactionalMap<String,WorkerState> workers = context.getMap(CURRENT_WORKERS);
		TransactionalList<WorkerState> availableWorkers = context.getList(AVAILABLE_WORKERS);


		workers.put(worker.getWorkerId(),worker);

		IAtomicReference<WorkerState> workerRef = h.getAtomicReference(worker.getWorkerId());
		workerRef.set(worker);


		if(!this.availableWorkers.contains(worker))
			availableWorkers.add(worker);

		
		
		context.commitTransaction();

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
		IAtomicReference<Job> jRef = h.getAtomicReference("job-" + j.getWorkerId());
		jRef.set(null);
		jobs.remove(j);

		WorkerState state = workers.get(j.getWorkerId());
		state.setAvailable(true);
		workers.put(state.getWorkerId(),state);
		
		IAtomicReference<WorkerState> workerRef = h.getAtomicReference(j.getWorkerId());
		workerRef.set(state);

	}

	@Override
	public void shutdown() {
		if(h != null)
			h.shutdown();

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
	public void jobDone(Job job) {
		try {
			clearJob(job);
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}

	@Override
	public boolean workerAvailable(String id) {
		return h.getLock(id).isLocked();
	}

	@Override
	public boolean isPretrain() {
		return (boolean) isPretrain.get();
	}

	@Override
	public void moveToFinetune() {
		isPretrain.set(false);
	}

	@Override
	public Job jobFor(String id) {
		IAtomicReference<Job> j = h.getAtomicReference("job-" + id);
		return j.get();
	}

	@Override
	public Job jobFor(WorkerState worker) {
		return jobFor(worker.getWorkerId());
	}

	@Override
	public void unlockWorker(String id) {
		TransactionContext context = beginTransaction();



		TransactionalList<WorkerState> availableWorkers = context.getList(AVAILABLE_WORKERS);
		TransactionalMap<String,WorkerState> workers = context.getMap(CURRENT_WORKERS);


		WorkerState w = workers.get(id);
		w.setAvailable(true);
		workers.put(w.getWorkerId(), w);

		if(!this.availableWorkers.contains(workers.get(id)))
			availableWorkers.add(workers.get(id));

		availableWorkers.add(workers.get(id));

		h.getAtomicReference("job-" + id).destroy();
		IAtomicReference<WorkerState> workerRef = h.getAtomicReference(id);
		workerRef.set(w);
		context.commitTransaction();
	}

	@Override
	public void lockWorker(String id) {
		WorkerState w = new WorkerState(id);
		w.setAvailable(false);
		workers.put(id, w);
		h.getAtomicReference(id).set(w);
	}


}
