package com.ccc.deeplearning.matrix.jblas.iterativereduce.actor.core.api;

import java.io.Serializable;


public interface EpochDoneListener<E> extends Serializable {

	void epochComplete(E result);
	
	void finish();
}
