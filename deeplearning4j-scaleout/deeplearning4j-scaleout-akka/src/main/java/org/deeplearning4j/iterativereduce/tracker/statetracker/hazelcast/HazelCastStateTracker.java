package org.deeplearning4j.iterativereduce.tracker.statetracker.hazelcast;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.deeplearning4j.iterativereduce.actor.core.Job;
import org.deeplearning4j.iterativereduce.actor.util.PortTaken;
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
import com.hazelcast.core.MemberAttributeEvent;
import com.hazelcast.core.MembershipEvent;
import com.hazelcast.core.MembershipListener;
/**
 * Tracks state of workers and jobs 
 * via hazelcast distributed data structures
 * @author Adam Gibson
 *
 */
public class HazelCastStateTracker implements StateTracker<UpdateableImpl> {

	/**
	 * 
	 */
	private static final long serialVersionUID = -7374372180080957334L;
	public final static String JOBS = "org.deeplearning4j.jobs";
	public final static String NUM_TIMES_PRETRAIN_RAN = "pretrainran";
	public final static String WORKERS = "org.deeplearning4j.workers";
	public final static String AVAILABLE_WORKERS = "AVAILABLE_WORKERS";
	public final static String NUM_TIMES_RUN_PRETRAIN = "PRETRAIN";
	public final static String TOPICS = "topics";
	public final static String RESULT = "RESULT";
	public final static String LOCKS = "LOCKS";
	public final static String HEART_BEAT = "heartbeat";
	public final static String IS_PRETRAIN = "ispretrain";
	public final static String RESULT_LOC = "RESULT_LOC";
	private volatile transient IAtomicReference<Object> master;
	private volatile transient IList<Job> jobs;
	private volatile transient IAtomicReference<Integer> numTimesPretrain;
	private volatile transient IAtomicReference<Integer> numTimesPretrainRan;

	private volatile transient IList<String> workers;
	private volatile  transient IList<String> topics;
	private volatile IAtomicReference<Object> isPretrain;
	private static Logger log = LoggerFactory.getLogger(HazelCastStateTracker.class);
	private transient Config config;
	public final static int DEFAULT_HAZELCAST_PORT = 2510;
	public final static String CURRENT_JOBS = "JOBS";
	private transient HazelcastInstance h;
	private String type;
	private int hazelCastPort = -1;

	private Map<String,Long> heartbeat;
	public HazelCastStateTracker() throws Exception {
		this(DEFAULT_HAZELCAST_PORT);

	}

	/**
	 * Initializes the state tracker binding to the given port
	 * @param stateTrackerPort the port to bind to
	 * @throws Exception
	 */
	public HazelCastStateTracker(int stateTrackerPort) throws Exception {
		this("master","master",stateTrackerPort);

	}
	
	public HazelCastStateTracker(String connectionString,String type,int stateTrackerPort) throws Exception {
		if(type.equals("master") && !PortTaken.portTaken(stateTrackerPort)) {
			this.hazelCastPort = stateTrackerPort;
			config = hazelcast();


			h = Hazelcast.newHazelcastInstance(config);
			h.getCluster().addMembershipListener(new MembershipListener() {

				@Override
				public void memberAdded(MembershipEvent membershipEvent) {
					log.info("Member added " + membershipEvent.toString());
				}

				@Override
				public void memberRemoved(MembershipEvent membershipEvent) {
					log.info("Member removed " + membershipEvent.toString());

				}

				@Override
				public void memberAttributeChanged(
						MemberAttributeEvent memberAttributeEvent) {
					log.info("Member changed " + memberAttributeEvent.toString());

				}

			});
		}
		else {
			log.info("Connecting to hazelcast on " + connectionString);
			ClientConfig client = new ClientConfig();
			client.getNetworkConfig().addAddress(connectionString);
			h = HazelcastClient.newHazelcastClient(client);

		}

		this.type = type;

		jobs = h.getList(JOBS);
		workers = h.getList(WORKERS);
		
		//we can make the assumption workers isn't empty because 
		//the master node by default comes with a set of workers
		if(!this.type.equals("master")) {
			while(workers.isEmpty()) {
				log.warn("Waiting for data sync...");
				Thread.sleep(1000);
			}
			
			log.info("Workers is " + workers.size());

		}

		topics = h.getList(TOPICS);
		heartbeat = h.getMap(HEART_BEAT);
		master = h.getAtomicReference(RESULT);
		isPretrain = h.getAtomicReference(IS_PRETRAIN);
		numTimesPretrain = h.getAtomicReference(NUM_TIMES_RUN_PRETRAIN);
		numTimesPretrainRan = h.getAtomicReference(NUM_TIMES_PRETRAIN_RAN);

		//set defaults only when master, otherwise, overrides previous values
		if(type.equals("master")) {
			numTimesPretrainRan.set(0);
			numTimesPretrain.set(1);
			isPretrain.set(true);
		}


	}

	private Config hazelcast() {
		Config conf = new Config();
		conf.getNetworkConfig().setPort(hazelCastPort);
		conf.getNetworkConfig().setPortAutoIncrement(false);



		conf.setProperty("hazelcast.initial.min.cluster.size","1");
		conf.setProperty("hazelcast.shutdownhook.enabled","false");

		JoinConfig join = conf.getNetworkConfig().getJoin();
		join.getMulticastConfig().setEnabled(true);
		join.getAwsConfig().setEnabled(false);
		join.getMulticastConfig().setEnabled(true);




		ListConfig jobConfig = new ListConfig();
		jobConfig.setName(JOBS);

		conf.addListConfig(jobConfig);



		ListConfig topicsConfig = new ListConfig();
		topicsConfig.setName(TOPICS);

		conf.addListConfig(topicsConfig);




		ListConfig availableWorkersConfig = new ListConfig();
		availableWorkersConfig.setName(AVAILABLE_WORKERS);
		conf.addListConfig(availableWorkersConfig);


		MapConfig heartbeatConfig = new MapConfig();
		heartbeatConfig.setName(HEART_BEAT);
		conf.addMapConfig(heartbeatConfig);


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
		return (boolean) isPretrain.get() && numTimesPretrainRan.get() < runPreTrainIterations();
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
		if(!workers.contains(id))
			workers.add(id);
	}

	@Override
	public List<String> jobIds() {
		List<String> ret = new ArrayList<>();
		for(Job j : this.jobs)
			ret.add(j.getWorkerId());
		return ret;
	}

	@Override
	public void addWorker(String worker) {
		log.info("Adding worker " + worker);
		heartbeat.put(worker, System.currentTimeMillis());
		if(!workers.contains(worker)) {
			workers.add(worker);
			log.info("Number of workers is now " + workers.size());

		}
	}

	@Override
	public void removeWorker(String worker) {
		workers.remove(worker);
	}

	@Override
	public List<String> workers() {
		return workers;
	}

	@Override
	public int numWorkers() {
		int num = workers.size();
		return num;
	}

	public synchronized HazelcastInstance getH() {
		return h;
	}

	public synchronized void setH(HazelcastInstance h) {
		this.h = h;
	}

	@Override
	public Map<String, Long> getHeartBeats() {
		return heartbeat;
	}

	@Override
	public void runPreTrainIterations(int numTimes) {
		numTimesPretrain.set(numTimes);
	}

	@Override
	public int runPreTrainIterations() {
		return numTimesPretrain.get();
	}

	@Override
	public int numTimesPreTrainRun() {
		return numTimesPretrainRan.get();
	}

	@Override
	public void incrementNumTimesPreTrainRan() {
		numTimesPretrainRan.set(numTimesPreTrainRun() + 1);		
	}




}
