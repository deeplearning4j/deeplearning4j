/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.scaleout.statetracker.hazelcast;


import com.hazelcast.client.HazelcastClient;
import com.hazelcast.client.config.ClientConfig;
import com.hazelcast.config.*;
import com.hazelcast.core.*;


import org.apache.commons.io.IOUtils;
import org.deeplearning4j.scaleout.actor.util.PortTaken;
import org.deeplearning4j.scaleout.aggregator.JobAggregator;
import org.deeplearning4j.scaleout.api.statetracker.*;
import org.deeplearning4j.scaleout.job.Job;
import org.deeplearning4j.scaleout.statetracker.updatesaver.LocalFileUpdateSaver;
import org.deeplearning4j.scaleout.statetracker.workretriever.LocalWorkRetriever;

import org.nd4j.linalg.io.ClassPathResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.net.InetAddress;
import java.util.*;

/**
 * Baseline hazelcast state tracker
 * @author Adam Gibson
 */

public abstract class BaseHazelCastStateTracker  implements StateTracker {

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
    public final static String BEST_LOSS = "bestloss";
    public final static String IMPROVEMENT_THRESHOLD = "improvementthreshold";
    public final static String EARLY_STOP = "earlystop";
    public final static String PATIENCE = "patience";
    public final static String BEGUN = "begun";
    public final static String NUM_BATCHES_SO_FAR_RAN = "numbatches";
    public final static String GLOBAL_REFERENCE = "globalreference";
    public final static String RECENTLY_CLEARED = "recentlycleared";

    private volatile transient IAtomicReference<Serializable> master;
    private volatile transient IList<Job> jobs;
    private volatile transient IAtomicReference<Integer> numTimesPretrain;
    private volatile transient IAtomicReference<Integer> numTimesPretrainRan;
    private volatile transient IAtomicReference<Double> bestLoss;
    private volatile transient IAtomicReference<Integer> numBatches;
    private volatile transient ISet<String> recentlyClearedJobs;

    private volatile transient IAtomicReference<Boolean> earlyStop;

    private volatile transient IMap<String,Serializable> references;
    private volatile transient IAtomicReference<Boolean> done;
    private volatile transient IList<String> replicate;
    private volatile transient IMap<String,Boolean> workerEnabled;
    private volatile transient IList<String> workers;
    private volatile  transient IList<String> topics;
    private volatile  transient IList<String> updates;
    private volatile IAtomicReference<Double> patience;
    private volatile IAtomicReference<Boolean> begunTraining;
    private volatile IAtomicReference<Integer> miniBatchSize;
    private WorkRetriever workRetriever = new LocalWorkRetriever();
    protected UpdateSaver saver;
    private volatile IAtomicReference<Boolean> isPretrain;
    private static final Logger log = LoggerFactory.getLogger(HazelCastStateTracker.class);
    private transient Config config;
    public final static int DEFAULT_HAZELCAST_PORT = 2510;
    private transient HazelcastInstance h;
    private String type = "master";
    private int hazelCastPort = -1;
    private String connectionString;
    private Map<String,Long> heartbeat;
    private StateTrackerDropWizardResource resource;
    protected JobAggregator jobAggregator;
    protected Serializable cachedCurrent;
    public final static String HAZELCAST_HOST = "hazelcast.host";
    private List<NewUpdateListener> listeners = new ArrayList<>();

    public BaseHazelCastStateTracker() throws Exception {
        this(DEFAULT_HAZELCAST_PORT);

    }

    @Override
    public <E extends Serializable> void define(String key, E o) {
        references.put(key,o);
    }

    @Override
    public <E extends Serializable> E get(String key) {
        return (E) references.get(key);
    }

    @Override
    public double count(String key) {
        IAtomicLong long2 = h.getAtomicLong(key);
        return long2.get();
    }

    @Override
    public void increment(String key, double by) {
        IAtomicLong long2 = h.getAtomicLong(key);
        long2.addAndGet((long) by);
    }

    @Override
    public void removeUpdateListener(NewUpdateListener listener) {
        listeners.remove(listener);
    }

    @Override
    public void addUpdateListener(NewUpdateListener listener) {
        listeners.add(listener);
    }

    /**
     * Number of batches ran so far
     *
     * @return the number of batches ran so far
     */
    @Override
    public int numBatchesRan() {
        return numBatches.get();
    }

    /**
     * Increments the number of batches ran.
     * This is purely a count and does not necessarily mean progress.
     *
     * @param numBatchesRan the number of batches ran to increment by
     */
    @Override
    public void incrementBatchesRan(int numBatchesRan) {
        numBatches.set(numBatchesRan + numBatches.get());
    }

    /**
     * Starts the rest api
     */
    @Override
    public void startRestApi() {
        String startApi = System.getProperty("startapi","false");
        Boolean b = Boolean.parseBoolean(startApi);
        if(!b)
            return;
        try {
            if(PortTaken.portTaken(8080) || PortTaken.portTaken(8180)) {
                log.warn("Port taken for rest api");
                return;
            }
            InputStream is = new ClassPathResource("/hazelcast/dropwizard.yml").getInputStream();


            resource = new StateTrackerDropWizardResource(this);
            File tmpConfig = new File("hazelcast/dropwizard.yml");
            if(!tmpConfig.getParentFile().exists())
                tmpConfig.getParentFile().mkdirs();
            BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(tmpConfig));
            IOUtils.copy(is, bos);
            bos.flush();


            resource.run("server",tmpConfig.getAbsolutePath());
            tmpConfig.deleteOnExit();
        }


        catch(Error e1) {
            log.warn("Unable to start server",e1);

        }
        catch(Exception e) {
            log.warn("Unable to start server",e);
        }

    }

    @Override
    public JobAggregator jobAggregator() {
        return jobAggregator;
    }

    @Override
    public void setJobAggregator(JobAggregator aggregator) {
        this.jobAggregator = aggregator;
    }

    public abstract UpdateSaver createUpdateSaver();

    /**
     * Current mini batch size
     *
     * @return
     */
    @Override
    public int miniBatchSize() {
        return miniBatchSize.get();
    }


    /**
     * Whether the cluster has begun training
     *
     * @return whether the cluster  has begun training
     */
    @Override
    public boolean hasBegun() {
        return begunTraining.get();
    }

    /**
     * Removes the worker data
     *
     * @param worker the worker to remove
     */
    @Override
    public void removeWorkerData(String worker) {
        workRetriever.clear(worker);
    }

    /**
     * The collection of dat
     *
     * @return
     */
    @Override
    public Collection<String> workerData() {
        return workRetriever.workers();
    }

    /**
     * Sets the work retriever to use for storing data sets for workers
     *
     * @param workRetriever the work retreiver to use with this state tracker
     */
    @Override
    public void setWorkRetriever(WorkRetriever workRetriever) {
        this.workRetriever = workRetriever;
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
    public void setUpdateSaver(UpdateSaver updateSaver) {
        this.saver = updateSaver;
    }

    /**
     * The update saver used with this state tracker
     *
     * @return the update saver used with this state tracker
     */
    @Override
    public UpdateSaver updateSaver() {
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
     * This means that each data applyTransformToDestination that is trained on
     * and loaded will be this batch size or lower
     * per worker
     *
     * @return the input split to use
     */
    @Override
    public int inputSplit() {
        Integer get = miniBatchSize.get();
        if(get == null)
            miniBatchSize.set(10);
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
     * this means that it will not iterate
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
    public void addUpdate(String id,Job update) {
        if(update == null)
            return;

        try {

            updateSaver().save(id,update);
            update.setWork(null);
            update.setResult(null);

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
    public abstract IterateAndUpdate updates();

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
    public BaseHazelCastStateTracker(int stateTrackerPort) throws Exception {
        this("master","master",stateTrackerPort);

    }

    /**
     * Worker constructor
     * @param connectionString
     */
    public BaseHazelCastStateTracker(String connectionString) throws Exception {
        this(connectionString,"worker",DEFAULT_HAZELCAST_PORT);
    }

    /**
     *
     * @param connectionString
     * @param type
     * @param stateTrackerPort
     * @throws Exception
     */
    public BaseHazelCastStateTracker(String connectionString,String type,int stateTrackerPort) throws Exception {
        log.info("Setting up hazelcast with type " + type + " connection string " + connectionString + " and port " + stateTrackerPort);


        if(type.equals("master") && !PortTaken.portTaken(stateTrackerPort)) {
            //sets up a proper connection string for reference wrt external actors needing a reference
            if(connectionString.equals("master")) {
                String hazelCastHost;
                try {
                    //try localhost fall back to 0.0.0.0
                    hazelCastHost = System.getProperty(HAZELCAST_HOST, InetAddress.getLocalHost().getHostName());
                }catch(Exception e) {
                    hazelCastHost = "0.0.0.0";
                }
                this.connectionString = hazelCastHost + ":" + stateTrackerPort;
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


        recentlyClearedJobs = h.getSet(RECENTLY_CLEARED);
        begunTraining = h.getAtomicReference(BEGUN);
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
        bestLoss = h.getAtomicReference(BEST_LOSS);
        earlyStop = h.getAtomicReference(EARLY_STOP);
        patience = h.getAtomicReference(PATIENCE);
        numBatches = h.getAtomicReference(NUM_BATCHES_SO_FAR_RAN);
        references = h.getMap(GLOBAL_REFERENCE);

        //applyTransformToDestination defaults only when master, otherwise, overrides previous values
        if(type.equals("master")) {
            begunTraining.set(false);
            saver = createUpdateSaver();
            numTimesPretrainRan.set(0);
            numTimesPretrain.set(1);
            isPretrain.set(true);
            done.set(false);
            resource = new StateTrackerDropWizardResource(this);
            bestLoss.set(Double.POSITIVE_INFINITY);
            earlyStop.set(true);
            numBatches.set(0);
        }

        workRetriever = new LocalWorkRetriever(h);



    }

    private Config hazelcast() {
        Config conf = new Config();
        conf.getNetworkConfig().setPort(hazelCastPort);
        conf.getNetworkConfig().setPortAutoIncrement(false);



        conf.setProperty("hazelcast.initial.min.cluster.size","1");
        conf.setProperty("hazelcast.shutdownhook.enabled","false");


        JoinConfig join = conf.getNetworkConfig().getJoin();

        boolean isAws = System.getProperty("hazelcast.aws","false").equals("true");
        log.info("Setting up Joiner with this being "  + (isAws ? "AWS" : "Multicast"));

        join.getAwsConfig().setEnabled(isAws);
        if(isAws) {
            join.getAwsConfig().setAccessKey(System.getProperty("hazelcast.access-key"));
            join.getAwsConfig().setSecretKey(System.getProperty("hazelcast.access-secret"));
        }
        join.getMulticastConfig().setEnabled(!isAws);


        String interf = System.getProperty("hazelcast.interface");
        if (interf != null) {
            conf.getNetworkConfig().getInterfaces().setEnabled(true).addInterface(interf);
        }

        ListConfig jobConfig = new ListConfig();
        jobConfig.setName(JOBS);

        conf.addListConfig(jobConfig);

        ListConfig replicateConfig = new ListConfig();
        replicateConfig.setName(REPLICATE_WEIGHTS);

        conf.addListConfig(replicateConfig);


        SetConfig cleared = new SetConfig();
        cleared.setName(RECENTLY_CLEARED);

        MapConfig referenceConfig = new MapConfig();
        referenceConfig.setName(GLOBAL_REFERENCE);
        conf.addMapConfig(referenceConfig);

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

        MapConfig workerEnabledConfig = new MapConfig();
        workerEnabledConfig.setName(WORKER_ENABLED);
        conf.addMapConfig(workerEnabledConfig);



        MapConfig fileUpdateSaver = new MapConfig();
        fileUpdateSaver.setName(LocalFileUpdateSaver.UPDATE_SAVER);
        conf.addMapConfig(fileUpdateSaver);


        MapConfig workRetriever = new MapConfig();
        workRetriever.setName(LocalWorkRetriever.WORK_RETRIEVER);
        conf.addMapConfig(workRetriever);

        return conf;

    }




    @Override
    public boolean addJobToCurrent(Job j) throws Exception {

        IAtomicReference<Job> r = h.getAtomicReference("job-" + j.workerId());


        if(r.get() != null || !r.isNull()) {
            boolean sent = false;
            while(!sent) {
                //always update
                for(String s : workers()) {
                    if(jobFor(s) == null) {
                        log.info("Redirecting worker " + j.workerId() + " to " + s + " due to work already being allocated");
                        r = h.getAtomicReference("job-" + s);
                        j.setWorkerId(s);
                        sent = true;
                    }
                }

            }
        }

        r.set(j);



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
        return jobs;
    }

    @Override
    public Set<String> recentlyCleared() {
        return recentlyClearedJobs;
    }

    /**
     * Assuming a job already exists, updates the job
     *
     * @param j the job to update
     */
    @Override
    public void updateJob(Job j) {
        IAtomicReference<Job> jRef = h.getAtomicReference("job-" + j.workerId());
        jRef.set(j);
    }

    @Override
    public void clearJob(String id) throws Exception {
        if(id == null) {
            log.warn("No job to clear; was null, returning");
            return;

        }

        recentlyClearedJobs.add(id);
        IAtomicReference<Job> jRef = h.getAtomicReference("job-" + id);
        if(jRef.isNull())
            return;
        jRef.clear();
        log.info("Destroyed job ref " + id);
        Job remove = null;
        for(Job j : jobs) {
            if(j.workerId().equals(id)) {
                remove = j;
                break;
            }
        }

        if(remove != null)
            jobs.remove(remove);

    }

    @Override
    public void shutdown() {
        if(h != null) {
            h.shutdown();
            h.getLifecycleService().shutdown();
        }
        if(resource != null)
            resource.shutdown();


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
    public  Serializable getCurrent() throws Exception {
        if(cachedCurrent != null)
            return cachedCurrent;
        Serializable u =  master.get();
        if(u == null)
            return null;
        return u;
    }

    @Override
    public  void setCurrent(Serializable e) throws Exception {
        if(e == null) {
            log.warn("Not setting a null update");
            return;
        }
        for(NewUpdateListener listener : listeners) {
            listener.onUpdate(e);
        }

        this.master.set(e);
    }



    @Override
    public Job jobFor(String id) {
        if(done.get())
            return null;
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
            ret.add(j.workerId());
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
        if(jobFor(worker) != null) {
            try {
                clearJob(worker);

            }catch(Exception e) {
                log.warn("Unable to clear job for worker with id" + worker);
            }
        }
    }

    @Override
    public List<String> workers() {
        return workers;
    }

    @Override
    public int numWorkers() {
        int num = workers.size();
        if(num < 1)
            throw new IllegalStateException("There appears to have been an issue during initialization. No workers found.");
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
        //reason being that isDone() may getFromOrigin called and throw errors
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
        //reason being that isDone() may getFromOrigin called and throw errors
        //this ensures a safe method call happens and just silently
        //returns true in case hazelcast is shutdown
        try {
            if(getCurrent() != null) {
                cachedCurrent = getCurrent();
                for(NewUpdateListener listener : listeners)
                    listener.onUpdate(cachedCurrent);
            }
            done.set(true);
            updateSaver().cleanup();

        }catch(Exception e) {
            log.warn("Hazelcast already shutdown...done() being called is pointless");
        }
    }

}
