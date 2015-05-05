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

package org.deeplearning4j.scaleout.api.statetracker;

import org.deeplearning4j.scaleout.aggregator.JobAggregator;
import org.deeplearning4j.scaleout.api.statetracker.IterateAndUpdate;
import org.deeplearning4j.scaleout.api.statetracker.UpdateSaver;
import org.deeplearning4j.scaleout.api.statetracker.WorkRetriever;
import org.deeplearning4j.scaleout.job.Job;

import java.io.Serializable;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Set;


/**
 * A StateTracker is a cross cluster 
 * monitor for state of workers and jobs
 * that workers have. This is mainly used
 * for tracking where training is at and 
 * for distribution of jobs across a cluster
 * relative to worker availability.
 *
 * @author Adam Gibson
 *
 */
public interface StateTracker extends Serializable {


    /**
     * The set of recently cleared jobs
     * @return the set of recently cleared jobs (based on worker ids)
     */
    Set<String> recentlyCleared();

    void increment(String key,double by);

    double count(String key);

    /**
     * Retrieve an element defined with define
     * @param key the key to use
     * @param <E>
     * @return the element defined or null
     */
    <E extends Serializable> E get(String key);

    /**
     * Define an object reference by key
     * @param key the key to use
     * @param o the object to define
     */
    <E extends Serializable> void  define(String key,E o);

    /**
     * Remove an update listener
     * @param listener the listener to remove
     */
    void removeUpdateListener(NewUpdateListener listener);

    /**
     * Add an update listener
     * @param listener a new update listener
     */
    void addUpdateListener(NewUpdateListener listener);


    JobAggregator jobAggregator();

    void setJobAggregator(JobAggregator aggregator);

    void setCurrent(Serializable e) throws Exception;

    /**
     * The current result
     * @return the current result
     * @throws Exception
     */
    Serializable getCurrent() throws Exception;

    /**
     * Removes the worker data
     * @param worker the worker to remove
     */
    void removeWorkerData(String worker);

    /**
     * The collection of data
     * @return the collection of workers who have data
     */
    Collection<String> workerData();

    /**
     * Sets the work retriever to use for storing data sets for workers
     * @param workRetriever the work retreiver to use with this state tracker
     */
    void setWorkRetriever(WorkRetriever workRetriever);

    /**
     * Loads the data for a given worker
     * @param workerId the worker id to load data for
     * @return the data applyTransformToDestination for a given worker
     */
    Job loadForWorker(String workerId);

    /**
     * Saves the data for the given worker to work on
     * @param workerId the worker to save
     * @param d the data for the worker
     */
    void saveWorker(String workerId, Job d);


    /**
     * A collection of worker updates.
     * This should be used to track
     * which workers have actually contributed an update for a given mini batch
     * @return the worker updates
     */
    Collection<String> workerUpdates();
    /**
     * The update saver to use
     * @param updateSaver the update saver to use
     */
    void setUpdateSaver(UpdateSaver updateSaver);

    /**
     * The update saver used with this state tracker
     * @return the update saver used with this state tracker
     */
    UpdateSaver updateSaver();

    /**
     * Assuming a job already exists, updates the job
     * @param j the job to update
     */
    void updateJob(Job j);

    /**
     * Sets the input split
     * @param inputSplit the input split to use
     */
    void setMiniBatchSize(int inputSplit);

    /**
     * The input split to use.
     * This means that each data applyTransformToDestination that is trained on
     * and loaded will be this batch size or lower
     * per worker
     * @return the input split to use
     */
    int inputSplit();

    /**
     * Returns the partition (optimal batch size)
     * given the available workers and the specified input split
     * @return the optimal batch size
     */
    int partition();

    /**
     * Returns the status of whether the worker is enabled or not
     * @param id the id of the worker to test
     * @return true if the worker is enabled, false otherwise
     */
    boolean workerEnabled(String id);

    /**
     * Enables the worker with the given id,
     * allowing it to take jobs again
     * @param id the id of the worker to enable
     */
    void enableWorker(String id);

    /**
     * Disables the worker with the given id,
     * this means that it will not iterate
     * or take any new jobs until re enabled
     * @param id the id of the worker to disable
     */
    void disableWorker(String id);

    /**
     * Updates the status of the worker to not needing replication
     * @param workerId the worker id to update
     */
    void doneReplicating(String workerId);

    /**
     * Adds a worker to the list to be replicate d
     * @param workerId the worker id to add
     */
    void addReplicate(String workerId);

    /**
     * Tracks worker ids that need state replication
     * @param workerId the worker id to replicate
     * @return the list of worker ids that need state replication
     */
    boolean needsReplicate(String workerId);

    /**
     * Adds an update to the current mini batch
     * @param id the id of the worker who did the update
     * @param update the update to add
     */
    void addUpdate(String id, Job update);

    /**
     * Updates  for mini batches
     * @return the current list of updates for mini batches
     */
    IterateAndUpdate updates();

    /**
     * Sets the connection string for connecting to the server
     * @param connectionString the connection string to use
     */
    void setConnectionString(String connectionString);

    /**
     * Connection string for connecting to the server
     * @return the connection string for connecting to the server
     */
    String connectionString();

    /**
     * Setter for the server port
     * @param port
     */
    void setServerPort(int port);


    /**
     * Starts the rest api
     */
    void startRestApi();

    /**
     * Gets the server port the state tracker is listening on (where applicable)
     * @return
     */
    int getServerPort();
    /**
     * Sets done to true
     */
    void finish();

    /**
     * Current mini batch size
     * @return
     */
    int miniBatchSize();

    /**
     * Whether the cluster is done training
     * @return whether the cluster is done training
     */
    boolean isDone();


    /**
     * Whether the cluster has begun training
     * @return whether the cluster  has begun training
     */
    boolean hasBegun();


    /**
     * Increments the number of times pre iterate has run.
     */
    void incrementNumTimesPreTrainRan();

    /**
     * Number of times pretrain has run so far
     * @return the number of times pretrain has run
     */
    int numTimesPreTrainRun();

    /**
     * Number of times to run pretrain
     * @param numTimes the number of times to run pretrain
     */
    void runPreTrainIterations(int numTimes);

    /**
     * Number of times to run pretrain
     * @return the number of times tp run pretrain
     */
    int runPreTrainIterations();



    /**
     * Current job ids
     * @return the curernt job ids
     */
    List<String> jobIds();

    /**
     * Adds a worker to the cluster,
     * also used for heartbeats.
     * This can be used by an external actor
     * to track if workers should be removed
     * based on heartbeat status
     * @param worker the worker to add or heartbeat with
     */
    void addWorker(String worker);
    /**
     * Removes the worker as a possible candidate for
     * job distribution in the cluster
     * @param worker
     */
    void removeWorker(String worker);
    /**
     * List of current workers
     * @return the list of current workers
     */
    List<String> workers();
    /**
     * The number of available workers
     *
     * @return the number of available workers
     */
    int numWorkers();


    /**
     * The heartbeats (timestamps)
     * of when workers last checked in to the cluster
     * @return the heartbeats for each worker
     */
    Map<String,Long> getHeartBeats();

    /**
     * The current job for a given worker
     * @param id the id of the worker to check
     * for a job on
     * @return the job for the worker or null
     */
    Job jobFor(String id);

    /**
     * Flags the given worker is available for work
     * @param id the worker to flag
     */
    void availableForWork(String id);

    /**
     * The list of current jobs
     * @return the list of current jobs
     * @throws Exception
     */
    List<Job> currentJobs() throws Exception;

    /**
     * Adds a topic (used for pub/sub) to the cluster
     * @param topic the topic to add
     * @throws Exception
     */
    void addTopic(String topic) throws Exception;

    /**
     * The list of available topics
     * @return the list of available topics
     * @throws Exception
     */
    List<String> topics() throws Exception;
    /**
     * Clears a job from the cluster
     * This should throw an exception when the job
     * doesn't exist
     * @param id the job to clear
     * @throws Exception
     */
    void clearJob(String id) throws Exception;
    /**
     * Attempts to add a job to the cluster
     * This should throw an exception when
     * the job being added to already exists
     * @param j the job to add
     * @return true if the job was added, false otherwise
     * @throws Exception
     */
    boolean addJobToCurrent(Job j) throws Exception;

    /**
     * Number of batches ran so far
     * @return the number of batches ran so far
     */
    int numBatchesRan();

    /**
     * Increments the number of batches ran.
     * This is purely a count and does not necessarily mean progress.
     * @param numBatchesRan the number of batches ran to increment by
     */
    void incrementBatchesRan(int numBatchesRan);

    /**
     * Shutsdown any connections on the cluster
     */
    void shutdown();

}
