package com.ccc.sendalyzeit.textanalytics.ml.scaleout.iterativereduce.jblas;



import com.ccc.sendalyzeit.textanalytics.ml.scaleout.iterativereduce.ComputableWorker;
/**
 * Worker for handling  subrows of a given set of matrix tasks.
 * Compute cycles are left to the user.
 * @author Adam Gibson
 *
 */
public abstract class ComputableWorkerMatrix implements ComputableWorker<UpdateableMatrix> {

	protected UpdateableMatrix workerMatrix;

	

	@Override
	public UpdateableMatrix getResults() {
		return workerMatrix;
	}

	@Override
	public void update(UpdateableMatrix t) {
		this.workerMatrix = t;
	}

	

}
