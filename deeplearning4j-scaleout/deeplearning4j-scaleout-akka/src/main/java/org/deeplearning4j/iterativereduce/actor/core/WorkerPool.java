package org.deeplearning4j.iterativereduce.actor.core;

import org.apache.commons.pool2.PooledObjectFactory;
import org.apache.commons.pool2.impl.AbandonedConfig;
import org.apache.commons.pool2.impl.GenericObjectPool;
import org.apache.commons.pool2.impl.GenericObjectPoolConfig;
import org.deeplearning4j.iterativereduce.actor.core.actor.WorkerState;

public class WorkerPool extends GenericObjectPool<WorkerState> {

	

	public WorkerPool(PooledObjectFactory<WorkerState> factory,
			GenericObjectPoolConfig config, AbandonedConfig abandonedConfig) {
		super(factory, config, abandonedConfig);
		// TODO Auto-generated constructor stub
	}

	public WorkerPool(PooledObjectFactory<WorkerState> factory) {
		super(factory, conf());
	}

	
	
	
	
	private static GenericObjectPoolConfig conf() {
		GenericObjectPoolConfig conf = new GenericObjectPoolConfig();
		conf.setBlockWhenExhausted(true);
		conf.setMaxTotal(Integer.MAX_VALUE);
		conf.setTestOnBorrow(false);
		conf.setTestOnCreate(false);
		conf.setMinEvictableIdleTimeMillis(-1);
	    return conf;
	}
}
