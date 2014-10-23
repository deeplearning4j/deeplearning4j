package org.deeplearning4j.iterativereduce.actor.core;

import java.io.Serializable;

public class ClearWorker implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 2861248705417593618L;
	private String id;
	public ClearWorker(String id) {
		super();
		this.id = id;
	}
	public  String getId() {
		return id;
	}
	public  void setId(String id) {
		this.id = id;
	}
	
	
	

}
