package org.deeplearning4j.iterativereduce.actor.util;

import java.util.ArrayList;
import java.util.List;

import akka.actor.ActorRef;
import akka.actor.ActorSystem;
import akka.cluster.Cluster;

public class ActorRefUtils {
	
	public static String absPath(ActorRef self,ActorSystem system) {
		String address = Cluster.get(system).selfAddress().toString();
		List<String> path2 = new ArrayList<String>();
		scala.collection.immutable.Iterable<String> elements = self.path().elements();
		scala.collection.Iterator<String> iter = elements.iterator();
		while(iter.hasNext())
			path2.add(iter.next());
		String absPath = "/" + org.apache.commons.lang3.StringUtils.join(path2, "/");
		return address + absPath + "/";
		
		
	}
}
