package com.ccc.sendalyzeit.textanalytics.ml.scaleout.iterativereduce.jblas;

import java.io.DataOutputStream;


import com.ccc.sendalyzeit.textanalytics.ml.scaleout.iterativereduce.ComputableMaster;


/**
 * Master results for a distributed computation for {@link org.jblas.DoubleMatrix}
 * @author Adam Gibson
 *
 */
public abstract class ComputableMasterMatrix implements ComputableMaster<UpdateableMatrix>{

	protected UpdateableMatrix masterMatrix;
	
	@Override
	public void complete(DataOutputStream ds) {
		masterMatrix.get().write(ds);
	}

	@Override
	public UpdateableMatrix getResults() {
		return masterMatrix;
	}

	

}
