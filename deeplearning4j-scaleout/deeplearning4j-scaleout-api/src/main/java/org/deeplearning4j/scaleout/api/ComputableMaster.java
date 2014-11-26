package org.deeplearning4j.scaleout.api;

import java.io.DataOutputStream;

import org.deeplearning4j.scaleout.conf.DeepLearningConfigurable;
import org.deeplearning4j.scaleout.job.Job;


/**
 * Master result applyTransformToDestination
 * Based on the iterative reduce specification seen here:
 * https://github.com/emsixteeen/IterativeReduce/blob/master/src/main/java/com/cloudera/iterativereduce/ComputableMaster.java
 * @author Adam Gibson
 *
 */
public interface ComputableMaster extends DeepLearningConfigurable {
	void complete(DataOutputStream ds);
	Job compute();
	Job getResults();
}
