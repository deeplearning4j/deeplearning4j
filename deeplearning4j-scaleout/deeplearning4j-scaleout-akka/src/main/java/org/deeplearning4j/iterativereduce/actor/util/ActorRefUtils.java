package org.deeplearning4j.iterativereduce.actor.util;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;

import org.deeplearning4j.scaleout.conf.Conf;
import org.deeplearning4j.scaleout.zookeeper.ZooKeeperConfigurationRegister;
import org.deeplearning4j.scaleout.zookeeper.ZooKeeperRunner;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import scala.concurrent.ExecutionContext;
import scala.concurrent.Future;

import akka.actor.ActorRef;
import akka.actor.ActorSystem;
import akka.cluster.Cluster;
import akka.dispatch.Futures;
import akka.dispatch.OnComplete;

public class ActorRefUtils {
	
	
	private static Logger log = LoggerFactory.getLogger(ActorRefUtils.class);
	
	/**
	 * Adds a shutdown hook for the system to shutdown
	 * with the jvm shutdown
	 * @param system the system to add a hook for
	 */
	public static void addShutDownForSystem(final ActorSystem system) {
		Runtime.getRuntime().addShutdownHook(new Thread(new Runnable() {

			@Override
			public void run() {
				Cluster cluster = Cluster.get(system);
				cluster.leave(cluster.selfAddress());
				system.shutdown();
			}

		}));
	}


	/**
	 * Starts an embedded zookeeper instance given an actor system
	 * @param system system to use
	 */
	public static void startEmbeddedZooKeeper(ActorSystem system) {
		System.setProperty("jute.maxbuffer","5048583");
		
		Future<Void> f = Futures.future(new Callable<Void>() {

			@Override
			public Void call() throws Exception {
				ZooKeeperRunner runner = new ZooKeeperRunner();
				runner.run();
				return null;
			}

		},system.dispatcher());
		
		throwExceptionIfExists(f,system.dispatcher());
	}

	/**
	 * Registers configuration with zookeeper, starts an embedded zookeeper
	 * if necessary
	 * @param conf conf to register
	 * @param system system to use
	 */
	public static void registerConfWithZooKeeper(final Conf conf,final ActorSystem system) {
		log.info("Stored master path of " + conf.getMasterAbsPath());
		Future<Void> f = Futures.future(new Callable<Void>() {

			@Override
			public Void call() throws Exception {
				log.info("Registering with zookeeper; if the logging stops here, ensure zookeeper is started");
				if(!PortTaken.portTaken(2181)) {
					log.info("No zookeeper found; starting an embedded zookeeper");
					startEmbeddedZooKeeper(system);
				}

				//register the configuration to zookeeper
				ZooKeeperConfigurationRegister reg = new ZooKeeperConfigurationRegister(conf,"master","localhost",2181);
				reg.register();
				reg.close();
				return null;
			}

		},system.dispatcher());

		f.onComplete(new OnComplete<Void>() {

			@Override
			public void onComplete(Throwable arg0, Void arg1) throws Throwable {
				if(arg0 != null)
					throw arg0;
				log.info("Registered conf with zookeeper");

			}

		}, system.dispatcher());

	}

	
	/**
	 * Returns the absolute path of the given actor given the system
	 * @param self the actor to getFromOrigin the absolute path for
	 * @param system the actor's system
	 * @return the absolute path of the given actor
	 */
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
	
	
	public static <T> void throwExceptionIfExists(Future<T> f,ExecutionContext context) {
		f.onComplete(new OnComplete<T>() {

			@Override
			public void onComplete(Throwable arg0, T arg1) throws Throwable {
				if(arg0 != null)
					throw arg0;
			}
			
		}, context);
	}
	
}
