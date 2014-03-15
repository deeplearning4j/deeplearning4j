package org.deeplearning4j.iterativereduce.tracker.statetracker;

import java.util.List;
import java.util.Map;

import org.deeplearning4j.iterativereduce.actor.core.Job;
import org.deeplearning4j.iterativereduce.actor.core.actor.WorkerState;

public interface StateTracker {

	
	
	
	List<Job> currentJobs() throws Exception;
	
	
	
	void clearJob(Job j) throws Exception;
	
	void addJobToCurrent(Job j) throws Exception;
	
	Map<String,WorkerState> currentWorkers() throws Exception;;
	
	WorkerState nextAvailableWorker() throws Exception;;
	
	void requeueJob(Job j) throws Exception;;
	
	void setWorkerDone(String id) throws Exception;;
	
	void clearWorker(WorkerState worker) throws Exception;;
	
	void addWorker(WorkerState worker) throws Exception;;

	boolean everyWorkerAvailable() throws Exception;;
	
	
	void shutdown();
	
}
