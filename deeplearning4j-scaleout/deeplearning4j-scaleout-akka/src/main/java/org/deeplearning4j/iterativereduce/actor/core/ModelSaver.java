package org.deeplearning4j.iterativereduce.actor.core;

import java.io.Serializable;
/**
 * Abstract idea of saving a model
 * @author Adam Gibson
 *
 */
public interface ModelSaver extends Serializable {
	/**
	 * Saves a serializable object
	 * @param ser the object to save
	 */
	void save(Serializable ser);
	
}
