package org.deeplearning4j.iterativereduce.actor.core;

import java.io.Serializable;

import org.deeplearning4j.scaleout.iterativereduce.Updateable;


public class MoreWorkMessage implements Serializable {

	public MoreWorkMessage(Updateable<?> updateable) {
		super();
		this.updateable = updateable;
	}

	/**
	 * 
	 */
	private static final long serialVersionUID = -4149080551476702735L;
	private Updateable<?> updateable;

	public synchronized Updateable<?> getUpdateable() {
		return updateable;
	}

	public synchronized void setUpdateable(Updateable<?> updateable) {
		this.updateable = updateable;
	}
	
	

}
