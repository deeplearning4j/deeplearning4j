package org.deeplearning4j.iterativereduce.tracker.statetracker.hazelcast;
import java.net.InetAddress;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;

import com.hazelcast.config.*;
import com.hazelcast.core.*;
import org.deeplearning4j.iterativereduce.actor.core.Job;
import org.deeplearning4j.iterativereduce.actor.util.PortTaken;
import org.deeplearning4j.iterativereduce.akka.DeepLearningAccumulator;
import org.deeplearning4j.iterativereduce.tracker.statetracker.IterateAndUpdate;
import org.deeplearning4j.iterativereduce.tracker.statetracker.StateTracker;
import org.deeplearning4j.iterativereduce.tracker.statetracker.UpdateSaver;
import org.deeplearning4j.scaleout.iterativereduce.multi.UpdateableImpl;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.hazelcast.client.HazelcastClient;
import com.hazelcast.client.config.ClientConfig;

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
    public final static String DONE = "done";
    public final static String UPDATES = "updates";
    public final static String REPLICATE_WEIGHTS = "replicate";
    public final static String HEART_BEAT = "heartbeat";
    public final static String WORKER_ENABLED = "workerenabled";
    public final static String INPUT_SPLIT = "inputsplit";
    public final static String IS_PRETRAIN = "ispretrain";
    private volatile transient IAtomicReference<Object> master;
    private volatile transient IList<Job> jobs;
    private volatile transient IAtomicReference<Integer> numTimesPretrain;
    private volatile transient IAtomicReference<Integer> numTimesPretrainRan;
    private volatile transient IAtomicReference<Boolean> done;
    private volatile transient IList<String> replicate;
    private volatile transient IMap<String,Boolean> workerEnabled;
    private volatile transient IList<String> workers;
    private volatile  transient IList<String> topics;
    private volatile  transient IList<String> updates;
    private volatile IAtomicReference<Integer> miniBatchSize;
    private UpdateSaver<UpdateableImpl> saver = new LocalFileUpdateSaver();
    private volatile IAtomicReference<Boolean> isPretrain;
    private static Logger log = LoggerFactory.getLogger(HazelCastStateTracker.class);
    private transient Config config;
    public final static int DEFAULT_HAZELCAST_PORT = 2510;
    private transient HazelcastInstance h;
    private String type = "master";
    private int hazelCastPort = -1;
    private String connectionString;
    private Map<String,Long> heartbeat;
    public HazelCastStateTracker() throws Exception {
        this(DEFAULT_HAZELCAST_PORT);

    }

    /**
     * A collection of worker updates.
     * This should be used to track
     * which workers have actually contributed an update for a given mini batch
     *
     * @return the worker updates
     */
    @Override
    public Collection<String> workerUpdates() {
        return updates;
    }

    /**
     * The update saver to use
     *
     * @param updateSaver the update saver to use
     */
    @Override
    public void setUpdateSaver(UpdateSaver<UpdateableImpl> updateSaver) {
        this.saver = updateSaver;
    }

    /**
     * The update saver used with this state tracker
     *
     * @return the update saver used with this state tracker
     */
    @Override
    public UpdateSaver<UpdateableImpl> updateSaver() {
        return saver;
    }

    /**
     * Sets the input split
     *
     * @param batchSize the input split to use
     */
    @Override
    public void setMiniBatchSize(int batchSize) {
        this.miniBatchSize.set(batchSize);
    }

    /**
     * The input split to use.
     * This means that each data set that is trained on
     * and loaded will be this batch size or lower
     * per worker
     *
     * @return the input split to use
     */
    @Override
    public int inputSplit() {
        return (miniBatchSize.get() * numWorkers()) / numWorkers();
    }

    /**
     * Returns the partition (optimal batch size)
     * given the available workers and the specified input split
     *
     * @return the optimal batch size
     */
    @Override
    public int partition() {
        return  inputSplit();
    }

    /**
     * Returns the status of whether the worker is enabled or not
     *
     * @param id the id of the worker to test
     * @return true if the worker is enabled, false otherwise
     */
    @Override
    public boolean workerEnabled(String id) {
        return workerEnabled.containsKey(id) && workerEnabled.get(id);
    }

    /**
     * Enables the worker with the given id,
     * allowing it to take jobs again
     *
     * @param id the id of the worker to enable
     */
    @Override
    public void enableWorker(String id) {
        workerEnabled.put(id,true);
    }

    /**
     * Disables the worker with the given id,
     * this means that it will not train
     * or take any new jobs until re enabled
     *
     * @param id the id of the worker to disable
     */
    @Override
    public void disableWorker(String id) {
        workerEnabled.put(id,false);
    }

    /**
     * Updates the status of the worker to not needing replication
     *
     * @param workerId the worker id to update
     */
    @Override
    public void doneReplicating(String workerId) {
        replicate.remove(workerId);
    }

    /**
     * Adds a worker to the list to be replicate d
     *
     * @param workerId the worker id to add
     */
    @Override
    public void addReplicate(String workerId) {
        if(!replicate.contains(workerId))
            replicate.add(workerId);
    }

    /**
     * Tracks worker ids that need state replication
     *
     * @param workerId the worker id to replicate
     * @return the list of worker ids that need state replication
     */
    @Override
    public boolean needsReplicate(String workerId) {
        return replicate.contains(workerId);
    }

    /**
     * Adds an update to the current mini batch
     * @param id the id of the worker who did the update
     * @param update the update to add
     */
    @Override
    public void addUpdate(String id,UpdateableImpl update) {
        try {
            updateSaver().save(id,update);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        updates.add(id);

    }

    /**
     * Updates  for mini batches
     *
     * @return the current list of updates for mini batches
     */
    @Override
    public IterateAndUpdate<UpdateableImpl> updates() {
        DeepLearningAccumulator d = new DeepLearningAccumulator(workerUpdates().size());
        DeepLearningAccumulatorIterateAndUpdate d2 = new DeepLearningAccumulatorIterateAndUpdate(d,updateSaver(),workers());
        return d2;
    }

    /**
     * Sets the connection string for connecting to the server
     *
     * @param connectionString the connection string to use
     */
    @Override
    public void setConnectionString(String connectionString) {
        this.connectionString = connectionString;
    }

    /**
     * Connection string for connecting to the server
     *
     * @return the connection string for connecting to the server
     */
    @Override
    public String connectionString() {
        return connectionString;
    }

    /**
     * Initializes the state tracker binding to the given port
     * @param stateTrackerPort the port to bind to
     * @throws Exception
     */
    public HazelCastStateTracker(int stateTrackerPort) throws Exception {
        this("master","master",stateTrackerPort);

    }

    /**
     * Worker constructor
     * @param connectionString
     */
    public HazelCastStateTracker(String connectionString) throws Exception {
        this(connectionString,"worker",DEFAULT_HAZELCAST_PORT);
    }

    public HazelCastStateTracker(String connectionString,String type,int stateTrackerPort) throws Exception {
        if(type.equals("master") && !PortTaken.portTaken(stateTrackerPort)) {
            //sets up a proper connection string for reference wrt external actors needing a reference
            if(connectionString.equals("master")) {
                String host = InetAddress.getLocalHost().getHostName();
                this.connectionString = host + ":" + stateTrackerPort;
            }

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

        else if(type.equals("master") && PortTaken.portTaken(stateTrackerPort))
            throw new IllegalStateException("Specified type was master and the port specified was taken, please specify a different port");

        else {

            setConnectionString(connectionString);
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

        miniBatchSize = h.getAtomicReference(INPUT_SPLIT);
        workerEnabled = h.getMap(WORKER_ENABLED);
        replicate = h.getList(REPLICATE_WEIGHTS);
        topics = h.getList(TOPICS);
        updates = h.getList(UPDATES);
        heartbeat = h.getMap(HEART_BEAT);
        master = h.getAtomicReference(RESULT);
        isPretrain = h.getAtomicReference(IS_PRETRAIN);
        numTimesPretrain = h.getAtomicReference(NUM_TIMES_RUN_PRETRAIN);
        numTimesPretrainRan = h.getAtomicReference(NUM_TIMES_PRETRAIN_RAN);
        done = h.getAtomicReference(DONE);


        //set defaults only when master, otherwise, overrides previous values
        if(type.equals("master")) {
            numTimesPretrainRan.set(0);
            numTimesPretrain.set(1);
            isPretrain.set(true);
            done.set(false);
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

        ListConfig replicateConfig = new ListConfig();
        replicateConfig.setName(REPLICATE_WEIGHTS);

        conf.addListConfig(replicateConfig);


        ListConfig topicsConfig = new ListConfig();
        topicsConfig.setName(TOPICS);

        conf.addListConfig(topicsConfig);


        ListConfig updatesConfig = new ListConfig();
        updatesConfig.setName(UPDATES);

        conf.addListConfig(updatesConfig);


        ListConfig availableWorkersConfig = new ListConfig();
        availableWorkersConfig.setName(AVAILABLE_WORKERS);
        conf.addListConfig(availableWorkersConfig);


        MapConfig heartbeatConfig = new MapConfig();
        heartbeatConfig.setName(HEART_BEAT);
        conf.addMapConfig(heartbeatConfig);

        MapConfig workerEnabledConifg = new MapConfig();
        workerEnabledConifg.setName(WORKER_ENABLED);
        conf.addMapConfig(workerEnabledConifg);

        return conf;

    }




    @Override
    public boolean addJobToCurrent(Job j) throws Exception {

        IAtomicReference<Job> r = h.getAtomicReference("job-" + j.getWorkerId());


        if(r.get() != null || !r.isNull()) {
            throw new IllegalArgumentException("Tried to add job with id " + j.getWorkerId() + " when one already exists");
        }

        r.set(j);

        //iterate over jobs without the work/data
        j.setWork(null);

        jobs.add(j);

        return true;

    }

    @Override
    public void setServerPort(int port) {
        this.hazelCastPort = port;
    }

    @Override
    public int getServerPort() {
        return hazelCastPort;
    }
    @Override
    public List<Job> currentJobs() throws Exception {
        return new ArrayList<>(jobs);
    }


    /**
     * Assuming a job already exists, updates the job
     *
     * @param j the job to update
     */
    @Override
    public void updateJob(Job j) {
        IAtomicReference<Job> jRef = h.getAtomicReference("job-" + j.getWorkerId());
        jRef.set(j);
    }

    @Override
    public void clearJob(String id) throws Exception {
        if(id == null) {
            log.warn("No job to clear; was null, returning");
            return;

        }

        IAtomicReference<Job> jRef = h.getAtomicReference("job-" + id);
        if(jRef.isNull())
            return;
        jRef.clear();
        log.info("Destroyed job ref " + id);
        Job remove = null;
        for(Job j : jobs) {
            if(j.getWorkerId().equals(id)) {
                remove = j;
                break;
            }
        }


        jobs.remove(remove);
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
        if(u == null)
            return null;
        return u.clone();
    }

    @Override
    public  void setCurrent(UpdateableImpl e) throws Exception {
        if(e == null || e.get() == null) {
            log.warn("Not setting a null update");
            return;
        }
        this.master.set(e);
    }


    @Override
    public boolean isPretrain() {
        return  isPretrain.get();
    }

    @Override
    public void moveToFinetune() {
        log.info("Moving to finetune");
        isPretrain.set(false);
    }

    @Override
    public Job jobFor(String id) {
        IAtomicReference<Job> j = h.getAtomicReference("job-" + id);
        if(j.isNull() || isCurrentlyJob(id))
            return null;
        return j.get();
    }

    private boolean isCurrentlyJob(String id) {
        for(Job j : jobs)
            if(j.equals(id))
                return true;
        return false;
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
        heartbeat.put(worker, System.currentTimeMillis());
        if(!workers.contains(worker)) {
            log.info("Adding worker " + worker);
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

    @Override
    public boolean isDone() {
        //reason being that isDone() may get called and throw errors
        //this ensures a safe method call happens and just silently
        //returns true in case hazelcast is shutdown
        try {
            return done.get();
        }catch(Exception e) {
            log.warn("Hazelcast already shutdown...returning true on isDone()");
            return true;
        }
    }

    @Override
    public void finish() {
        //reason being that isDone() may get called and throw errors
        //this ensures a safe method call happens and just silently
        //returns true in case hazelcast is shutdown
        try {
            done.set(true);
        }catch(Exception e) {
            log.warn("Hazelcast already shutdown...done() being set is pointless");
        }
    }




}
