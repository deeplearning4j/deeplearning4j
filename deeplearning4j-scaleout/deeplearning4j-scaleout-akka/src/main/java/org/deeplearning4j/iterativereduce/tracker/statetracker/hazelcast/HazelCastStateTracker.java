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
import com.hazelcast.core.IMap;

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
	private volatile IList<String> ids;

	private volatile  IList<String> topics;
	private volatile IAtomicReference<Object> isPretrain;
	private static Logger log = LoggerFactory.getLogger(HazelCastStateTracker.class);
	private Config config;
	public final static int DEFAULT_HAZELCAST_PORT = 2510;
	public final static String CURRENT_JOBS = "JOBS";
	private HazelcastInstance h;
	private String type;
	public HazelCastStateTracker() throws Exception {
		this("localhost:2181","master");

	}

	public HazelCastStateTracker(String connectionString,String type) throws Exception {
		if(type.equals("master")) {
			config = hazelcast();
			h = Hazelcast.newHazelcastInstance(config);

		}
		else {
			ClientConfig client = new ClientConfig();
			client.getNetworkConfig().setAddresses(Arrays.asList(connectionString));
			h = HazelcastClient.newHazelcastClient(client);

		}

		this.type = type;

		jobs = h.getList(JOBS);
		ids = h.getList(CURRENT_WORKERS);
		topics = h.getList(TOPICS);
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




	@Override
	public boolean addJobToCurrent(Job j) throws Exception {
		IAtomicReference<Job> r = h.getAtomicReference("job-" + j.getWorkerId());
		if(r.get() != null) {
			log.info("Currently locked unable to add job for current worker");
			return false;
		}



		jobs.add(j);
		r.set(j);

		return true;

	}


	@Override
	public List<Job> currentJobs() throws Exception {
		return new ArrayList<Job>(jobs);
	}




	@Override
	public void clearJob(Job j) throws Exception {
		IAtomicReference<Job> jRef = h.getAtomicReference("job-" + j.getWorkerId());
		jRef.destroy();
		jobs.remove(j);

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
	public void availableForWork(String id) {
		if(!ids.contains(id))
			ids.add(id);
	}

	@Override
	public List<String> jobIds() {
		return ids;
	}




}
