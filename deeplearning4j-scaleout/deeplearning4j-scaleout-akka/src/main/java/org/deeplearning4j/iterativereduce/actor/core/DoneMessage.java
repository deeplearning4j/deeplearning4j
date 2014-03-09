package org.deeplearning4j.iterativereduce.actor.core;

import java.io.Serializable;
/**
 * Sent to the master actor when there is no more work
 * @author Adam Gibson
 *
 */
public class DoneMessage implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = -6670506591283846265L;
	private static DoneMessage INSTANCE = new DoneMessage();
	private DoneMessage() {}
	
	public static DoneMessage getInstance() {
		return INSTANCE;
	}

}
