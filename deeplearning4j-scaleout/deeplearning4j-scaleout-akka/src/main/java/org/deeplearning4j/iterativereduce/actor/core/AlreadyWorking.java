package org.deeplearning4j.iterativereduce.actor.core;

import java.io.Serializable;

public class AlreadyWorking implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = -597971431388818083L;
	private String id;
	public  String getId() {
		return id;
	}
	public  void setId(String id) {
		this.id = id;
	}
	public AlreadyWorking(String id) {
		super();
		this.id = id;
	}
	public AlreadyWorking() {
		super();
	}
	
	
	
	

}
