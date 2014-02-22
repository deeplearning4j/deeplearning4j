package org.deeplearning4j.iterativereduce.actor.core.api;

import java.io.Serializable;


public interface EpochDoneListener<E> extends Serializable {

	void epochComplete(E result);
	
	void finish();
}
