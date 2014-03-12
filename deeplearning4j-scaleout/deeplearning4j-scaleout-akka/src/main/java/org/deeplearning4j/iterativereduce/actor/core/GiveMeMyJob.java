package org.deeplearning4j.iterativereduce.actor.core;

import java.io.Serializable;

public class GiveMeMyJob implements Serializable {

	private String id;
	private Job job;
	public GiveMeMyJob(String id,Job job) {
		super();
		this.id = id;
		this.job = job;
	}

	public  String getId() {
		return id;
	}

	public  void setId(String id) {
		this.id = id;
	}

	public synchronized Job getJob() {
		return job;
	}

	public synchronized void setJob(Job job) {
		this.job = job;
	}
	
	
	
}
