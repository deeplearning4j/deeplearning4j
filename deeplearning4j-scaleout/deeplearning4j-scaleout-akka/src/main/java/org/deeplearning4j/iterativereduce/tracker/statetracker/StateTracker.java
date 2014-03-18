package org.deeplearning4j.iterativereduce.tracker.statetracker;

import java.io.Serializable;
import java.util.List;

import org.deeplearning4j.iterativereduce.actor.core.Job;
import org.deeplearning4j.scaleout.iterativereduce.Updateable;

public interface StateTracker<E extends Updateable<?>> extends Serializable {

	
	boolean isPretrain();
	void moveToFinetune();
	
	
	List<String> jobIds();
	
	public Job jobFor(String id);
	
	
	void availableForWork(String id);
	
	
	List<Job> currentJobs() throws Exception;
	
	
	void addTopic(String topic) throws Exception;
	
	List<String> topics() throws Exception;
	
	void clearJob(Job j) throws Exception;
	
	boolean addJobToCurrent(Job j) throws Exception;

	E getCurrent() throws Exception;
	void setCurrent(E e) throws Exception;
	
	
	void jobDone(Job job);
	
	
	void shutdown();
	
}
