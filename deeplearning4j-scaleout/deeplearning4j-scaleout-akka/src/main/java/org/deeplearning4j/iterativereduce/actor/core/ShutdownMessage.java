package org.deeplearning4j.iterativereduce.actor.core;

import java.io.Serializable;

/**
 * Message sent to the cluster when 
 * an actor system on the cluster should be shutdown.
 * This handles the do a lot of work then die pattern.
 * @author Adam Gibson
 *
 */
public class ShutdownMessage implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = -6903668251648532497L;

	

}
