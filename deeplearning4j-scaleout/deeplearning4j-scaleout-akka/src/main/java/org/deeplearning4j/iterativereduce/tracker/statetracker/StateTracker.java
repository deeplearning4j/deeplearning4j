package org.deeplearning4j.iterativereduce.tracker.statetracker;

import java.io.Serializable;
import java.util.Collection;
import java.util.List;
import java.util.Map;

import org.deeplearning4j.iterativereduce.actor.core.Job;
import org.deeplearning4j.linalg.dataset.DataSet;
import org.deeplearning4j.nn.BaseMultiLayerNetwork;
import org.deeplearning4j.optimize.api.TrainingEvaluator;
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
     * Removes the worker data
     * @param worker the worker to remove
     */
    public void removeWorkerData(String worker);

    /**
     * The collection of dat
     * @return the collection of workers who have data
     */
    public Collection<String> workerData();

    /**
     * Sets the work retriever to use for storing data sets for workers
     * @param workRetriever the work retreiver to use with this state tracker
     */
    public void setWorkRetriever(WorkRetriever workRetriever);

    /**
     * Loads the data for a given worker
     * @param workerId the worker id to load data for
     * @return the data applyTransformToDestination for a given worker
     */
    public DataSet loadForWorker(String workerId);

    /**
     * Saves the data for the given worker to work on
     * @param workerId the worker to save
     * @param d the data for the worker
     */
   public void saveWorker(String workerId,DataSet d);


    /**
     * Creates a training evaluator using the given neural network
     * @param network the neural network to use
     * @return a training evaluator based on the configuration of the state tracker
     * and the given network
     */
    public TrainingEvaluator create(BaseMultiLayerNetwork network);

    /**
     * Set the data applyTransformToDestination cache to use for fetching the test applyTransformToDestination
     * @param cache the cache to use
     */
    public void setDataSetCache(DataSetCache cache);

    /**
     * The patience improvement to use
     * @param improvmentThreshold the patience improvement to applyTransformToDestination
     */
    public void setImprovmentThreshold(double improvmentThreshold);

    /**
     * How much to bump up the patience wrt training improvements
     * for early stopping
     * @return the current patience improvement
     */
    public double improvmentThreshold();

    /**
     * Setter for patience
     * @param patience
     */
    public void setPatience(double patience);

    /**
     * Patience is what controls early stopping
     * @return the patience for the trainer
     */
    public double patience();


    /**
     * Improvement threshold for early stopping, aka
     * the minimum
     * @return
     */
    public double improvementThreshold();

    /**
     * The test applyTransformToDestination to use for validation
     * @return the test to use for validation
     */
    public DataSet testSet();

    /**
     * Sets the best loss
     * @param bestLoss the best loss to use
     */
    public void setBestLoss(double bestLoss);

    /**
     * The best validation loss so far
     * @return the best validation loss so far
     */
    public double bestLoss();

    /**
     * The number of epochs to test on
     * @return the number of epochs to test on
     */
    public int validationEpochs();

    /**
     * Whether to validate against a held out test applyTransformToDestination and test for validation error.
     *
     * @return whether to validate against a held out test applyTransformToDestination and test for validation error.
     */
    public boolean isEarlyStopTesting();

    /**
     * A collection of worker updates.
     * This should be used to track
     * which workers have actually contributed an update for a given mini batch
     * @return the worker updates
     */
    public Collection<String> workerUpdates();
    /**
     * The update saver to use
     * @param updateSaver the update saver to use
     */
    public void setUpdateSaver(UpdateSaver<E> updateSaver);

    /**
     * The update saver used with this state tracker
     * @return the update saver used with this state tracker
     */
    public UpdateSaver<E> updateSaver();

    /**
     * Assuming a job already exists, updates the job
     * @param j the job to update
     */
    public void updateJob(Job j);

    /**
     * Sets the input split
     * @param inputSplit the input split to use
     */
    public void setMiniBatchSize(int inputSplit);

    /**
     * The input split to use.
     * This means that each data applyTransformToDestination that is trained on
     * and loaded will be this batch size or lower
     * per worker
     * @return the input split to use
     */
    public int inputSplit();

    /**
     * Returns the partition (optimal batch size)
     * given the available workers and the specified input split
     * @return the optimal batch size
     */
    public int partition();

    /**
     * Returns the status of whether the worker is enabled or not
     * @param id the id of the worker to test
     * @return true if the worker is enabled, false otherwise
     */
    public boolean workerEnabled(String id);

    /**
     * Enables the worker with the given id,
     * allowing it to take jobs again
     * @param id the id of the worker to enable
     */
    public void enableWorker(String id);

    /**
     * Disables the worker with the given id,
     * this means that it will not iterate
     * or take any new jobs until re enabled
     * @param id the id of the worker to disable
     */
    public void disableWorker(String id);

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
     * @param id the id of the worker who did the update
     * @param update the update to add
     */
    public void addUpdate(String id,E update);

    /**
     * Updates  for mini batches
     * @return the current list of updates for mini batches
     */
    public IterateAndUpdate<E> updates();

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
     * Starts the rest api
     */
    void startRestApi();

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
     * Begin training
     */
    void beginTraining();

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
