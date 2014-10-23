package org.deeplearning4j.iterativereduce.actor.util;

import java.net.ServerSocket;

public class PortTaken {

	public static boolean portTaken(int port) {
		try {
			ServerSocket s = new ServerSocket(port);
			s.close();
			return false;
		}catch(Exception e) {
			return true;
		}

	}

}
