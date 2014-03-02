package org.deeplearning4j.scaleout.iterativereduce.multi.gradient;

import java.io.DataOutputStream;

import org.deeplearning4j.scaleout.iterativereduce.ComputableMaster;




/**
 * Master results for a distributed computation for {@link org.jblas.DoubleMatrix}
 * @author Adam Gibson
 *
 */
public abstract class ComputableMasterImpl implements ComputableMaster<UpdateableGradientImpl>{

	protected UpdateableGradientImpl masterResults;
	
	@Override
	public void complete(DataOutputStream ds) {
		masterResults.get().write(ds);
	}

	@Override
	public UpdateableGradientImpl getResults() {
		return masterResults;
	}

	

}
