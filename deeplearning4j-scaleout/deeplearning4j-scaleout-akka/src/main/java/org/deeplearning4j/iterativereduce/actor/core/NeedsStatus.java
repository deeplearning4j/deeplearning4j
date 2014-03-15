package org.deeplearning4j.iterativereduce.actor.core;

import java.io.Serializable;

public class NeedsStatus implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = -3392823043790129100L;
	private static NeedsStatus INSTANCE = new NeedsStatus();
	
	
	private NeedsStatus() {}
	
	public static NeedsStatus getInstance() {
		return INSTANCE;
	}
}
