package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda.matrix.jblas.iterativereduce.actor;

import java.io.Serializable;

import com.ccc.sendalyzeit.textanalytics.ml.scaleout.iterativereduce.jblas.UpdateableMatrix;

public class UpdateMessage implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = -52064549181572354L;
	private UpdateableMatrix updateable;
	public UpdateMessage(UpdateableMatrix updateable) {
		super();
		this.updateable = updateable;
	}
	public UpdateableMatrix getUpdateable() {
		return updateable;
	}
	
	
	

}
