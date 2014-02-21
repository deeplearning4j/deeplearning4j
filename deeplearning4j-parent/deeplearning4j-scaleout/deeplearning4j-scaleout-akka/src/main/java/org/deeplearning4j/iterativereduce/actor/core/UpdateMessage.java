package org.deeplearning4j.iterativereduce.actor.core;

import java.io.Serializable;

import org.deeplearning4j.scaleout.iterativereduce.Updateable;


public class UpdateMessage<E> implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = -52064549181572354L;
	private Updateable<E> updateable;
	public UpdateMessage(Updateable<E> updateable) {
		super();
		this.updateable = updateable;
	}
	public Updateable<E> getUpdateable() {
		return updateable;
	}
	
	
	

}
