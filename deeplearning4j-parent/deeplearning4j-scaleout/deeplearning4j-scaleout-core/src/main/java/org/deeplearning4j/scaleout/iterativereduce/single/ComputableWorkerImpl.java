package org.deeplearning4j.scaleout.iterativereduce.single;



import org.deeplearning4j.scaleout.iterativereduce.ComputableWorker;
/**
 * Worker for handling  subrows of a given set of matrix tasks.
 * Compute cycles are left to the user.
 * @author Adam Gibson
 *
 */
public abstract class ComputableWorkerImpl implements ComputableWorker<UpdateableSingleImpl> {

	protected UpdateableSingleImpl workerMatrix;

	

	@Override
	public UpdateableSingleImpl getResults() {
		return workerMatrix;
	}

	@Override
	public void update(UpdateableSingleImpl t) {
		this.workerMatrix = t;
	}

	

}
