package org.deeplearning4j.iterativereduce.tracker.statetracker;

import java.io.Serializable;
import java.util.List;
import java.util.Map;

import org.deeplearning4j.iterativereduce.actor.core.Job;
import org.deeplearning4j.scaleout.iterativereduce.Updateable;
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
 * @param <E>
 */
public interface StateTracker<E extends Updateable<?>> extends Serializable {


    /**
     * Updates the status of the worker to not needing replication
     * @param workerId the worker id to update
     */
    public void doneReplicating(String workerId);

    /**
     * Adds a worker to the list to be replicate d
     * @param workerId the worker id to add
     */
    public void addReplicate(String workerId);

    /**
     * Tracks worker ids that need state replication
     * @param workerId the worker id to replicate
     * @return the list of worker ids that need state replication
     */
    public boolean needsReplicate(String workerId);

    /**
     * Adds an update to the current mini batch
     * @param update the update to add
     */
    public void addUpdate(E update);

    /**
     * Updates  for mini batches
     * @return the current list of updates for mini batches
     */
    public List<E> updates();

    /**
     * Sets the connection string for connecting to the server
     * @param connectionString the connection string to use
     */
    public void setConnectionString(String connectionString);

    /**
     * Connection string for connecting to the server
     * @return the connection string for connecting to the server
     */
    public String connectionString();

    /**
     * Setter for the server port
     * @param port
     */
    public void setServerPort(int port);

    /**
     * Gets the server port the state tracker is listening on (where applicable)
     * @return
     */
    public int getServerPort();
	/**
	 * Sets done to true
	 */
	void finish();
	
	/**
	 * Whether the cluster is done training
	 * @return whether the cluster is done training
	 */
	boolean isDone();
	
	/**
	 * Increments the number of times pre train has run.
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
	 * Whether the training is currently
	 * in pretrain mode
	 * @return whether the training is currently in pretrain mode
	 */
	boolean isPretrain();
	
	/**
	 * Move from pretrain phase to finetune phase
	 */
	void moveToFinetune();
	
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
	public Job jobFor(String id);
	
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
	 * @param j the job to clear
	 * @throws Exception
	 */
	void clearJob(Job j) throws Exception;
	/**
	 * Attempts to add a job to the cluster
	 * @param j the job to add
	 * @return true if the job was added, false otherwise
	 * @throws Exception
	 */
	boolean addJobToCurrent(Job j) throws Exception;
	
	/**
	 * Gets the current result
	 * @return the current results
	 * @throws Exception
	 */
	E getCurrent() throws Exception;
	/**
	 * Updates the current result
	 * @param e the current result
	 * @throws Exception
	 */
	void setCurrent(E e) throws Exception;
	
	/**
	 * Marks the job as done
	 * @param job the job to mark
	 */
	void jobDone(Job job);
	
	/**
	 * Shutsdown any connections on the cluster
	 */
	void shutdown();
	
}
