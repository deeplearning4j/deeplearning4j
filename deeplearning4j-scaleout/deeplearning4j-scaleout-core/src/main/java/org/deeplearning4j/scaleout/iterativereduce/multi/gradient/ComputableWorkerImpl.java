package org.deeplearning4j.scaleout.iterativereduce.multi.gradient;



import org.deeplearning4j.scaleout.iterativereduce.ComputableWorker;
/**
 * Worker for handling  subrows of a given set of matrix tasks.
 * Compute cycles are left to the user.
 * @author Adam Gibson
 *
 */
public abstract class ComputableWorkerImpl implements ComputableWorker<UpdateableGradientImpl> {

	protected UpdateableGradientImpl workerResult;

	

	@Override
	public UpdateableGradientImpl getResults() {
		return workerResult;
	}

	@Override
	public void update(UpdateableGradientImpl t) {
		this.workerResult = t;
	}

	

}
