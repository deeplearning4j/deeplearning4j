package org.deeplearning4j.iterativereduce.actor.core;

import java.io.Serializable;

public class JobFailed implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = -5416932480720155131L;
	private Job failed;
	public JobFailed(Job failed) {
		super();
		this.failed = failed;
	}
	public Job getFailed() {
		return failed;
	}
	public void setFailed(Job failed) {
		this.failed = failed;
	}
	
	

}
