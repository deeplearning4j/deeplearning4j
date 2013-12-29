package com.ccc.deeplearning.matrix.jblas.iterativereduce.actor.core;

import java.io.Serializable;

import com.ccc.deeplearning.scaleout.iterativereduce.Updateable;

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
