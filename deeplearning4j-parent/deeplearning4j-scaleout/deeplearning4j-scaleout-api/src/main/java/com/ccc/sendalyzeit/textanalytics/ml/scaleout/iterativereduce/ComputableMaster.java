package com.ccc.sendalyzeit.textanalytics.ml.scaleout.iterativereduce;

import java.io.DataOutputStream;
import java.util.Collection;

import com.ccc.sendalyzeit.textanalytics.ml.scaleout.conf.Conf;

/**
 * Master result set
 * Based on the iterative reduce specification seen here:
 * https://github.com/emsixteeen/IterativeReduce/blob/master/src/main/java/com/cloudera/iterativereduce/ComputableMaster.java
 * @author Adam Gibson
 *
 * @param <RECORD_TYPE> probably a matrix or vector
 */
public interface ComputableMaster<RECORD_TYPE extends Updateable<?>> {
    void setup(Conf conf);
	void complete(DataOutputStream ds);
	RECORD_TYPE compute(Collection<RECORD_TYPE> workerUpdates,Collection<RECORD_TYPE> masterUpdates);
	RECORD_TYPE getResults();
}
