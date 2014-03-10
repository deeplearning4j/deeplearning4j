package org.deeplearning4j.iterativereduce.actor.core;

import java.io.Serializable;

public class Ack implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = -6012261296646486276L;

	private static Ack INSTANCE = new Ack();
	private Ack() {}
	
	public static Ack getInstance() {
		return INSTANCE;
	}
}
