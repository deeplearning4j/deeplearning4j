package org.deeplearning4j.scaleout.iterativereduce;

import java.util.List;

import org.deeplearning4j.scaleout.conf.Conf;

/**
 * Shamelessly based on iterative reduce work doen by cloudera: 
 * https://github.com/emsixteeen/IterativeReduce/blob/master/src/main/java/com/cloudera/iterativereduce/ComputableWorker.java
 * @author Adam Gibson
 *
 * @param <RECORD_TYPE> a training example: probably a vector or matrix
 */
public interface ComputableWorker<RECORD_TYPE extends Updateable<?>> {

	
	void setup(Conf conf);
	
	RECORD_TYPE compute(List<RECORD_TYPE> records);
	
	RECORD_TYPE compute();
	
	RECORD_TYPE getResults();
	
	void update(RECORD_TYPE t);
	
	public boolean incrementIteration();
}
