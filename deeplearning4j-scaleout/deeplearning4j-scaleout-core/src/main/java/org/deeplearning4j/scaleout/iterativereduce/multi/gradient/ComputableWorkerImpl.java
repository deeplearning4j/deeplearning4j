package org.deeplearning4j.scaleout.iterativereduce.multi.gradient;



import org.deeplearning4j.scaleout.iterativereduce.ComputableWorker;
import org.deeplearning4j.scaleout.iterativereduce.multi.UpdateableImpl;
/**
 * Worker for handling  subrows of a given set of matrix tasks.
 * Compute cycles are left to the user.
 * @author Adam Gibson
 *
 */
public abstract class ComputableWorkerImpl implements ComputableWorker<UpdateableImpl> {

	protected UpdateableImpl workerResult;

	

	@Override
	public UpdateableImpl getResults() {
		return workerResult;
	}

	@Override
	public void update(UpdateableImpl t) {
		this.workerResult = t;
	}

	

}
