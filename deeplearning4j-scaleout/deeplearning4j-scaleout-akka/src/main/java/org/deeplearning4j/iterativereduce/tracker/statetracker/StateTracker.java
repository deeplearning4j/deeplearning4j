package org.deeplearning4j.iterativereduce.tracker.statetracker;

import java.io.Serializable;
import java.util.List;
import java.util.Map;

import org.deeplearning4j.iterativereduce.actor.core.Job;
import org.deeplearning4j.iterativereduce.actor.core.actor.WorkerState;
import org.deeplearning4j.scaleout.iterativereduce.Updateable;

public interface StateTracker<E extends Updateable<?>> extends Serializable {

	
	void jobRequeued(Job j);
	
	List<Job> currentJobs() throws Exception;
	
	List<Job> jobsToRedistribute();
	
	void addTopic(String topic) throws Exception;
	
	List<String> topics() throws Exception;
	
	void clearJob(Job j) throws Exception;
	
	void addJobToCurrent(Job j) throws Exception;
	
	Map<String,WorkerState> currentWorkers() throws Exception;
	
	WorkerState nextAvailableWorker() throws Exception;;
	
	void requeueJob(Job j) throws Exception;;
	
	void setWorkerDone(String id) throws Exception;;
	
	void clearWorker(WorkerState worker) throws Exception;;
	
	void addWorker(WorkerState worker) throws Exception;;

	boolean everyWorkerAvailable() throws Exception;;
	
	
	boolean workerAvailable(String id);
	
	
	E getCurrent() throws Exception;
	void setCurrent(E e) throws Exception;
	
	
	void jobDone(Job job);
	
	
	void shutdown();
	
}
