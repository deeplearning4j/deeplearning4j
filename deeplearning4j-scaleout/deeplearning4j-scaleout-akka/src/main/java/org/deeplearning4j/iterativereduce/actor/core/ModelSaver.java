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
	void save(Serializable ser) throws Exception;

    /**
     * Loads the model from the location that it saves to
     * @param type to use when loading
     */
    <E> E load(Class<E> type);

    /**
     * Returns whether a model exists or not
     * @return true if the model exists, false otherwise
     */
    boolean exists();
	
}
