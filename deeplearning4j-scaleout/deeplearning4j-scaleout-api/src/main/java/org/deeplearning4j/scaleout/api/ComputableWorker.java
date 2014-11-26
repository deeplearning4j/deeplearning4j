package org.deeplearning4j.scaleout.api;

import java.util.List;

import org.deeplearning4j.scaleout.conf.DeepLearningConfigurable;
import org.deeplearning4j.scaleout.job.Job;

/**
 * Shamelessly based on iterative reduce work done by cloudera:
 * https://github.com/emsixteeen/IterativeReduce/blob/master/src/main/java/com/cloudera/iterativereduce/ComputableWorker.java
 * @author Adam Gibson
 *
 */
public interface ComputableWorker extends DeepLearningConfigurable {

	

	Job compute(List<Job> records);

    Job compute();

    Job getResults();
	
	void update(Job t);
	
	public boolean incrementIteration();
}
