package com.ccc.deeplearning.scaleout.iterativereduce.single;

import java.io.DataOutputStream;


import com.ccc.deeplearning.scaleout.iterativereduce.ComputableMaster;


/**
 * Master results for a distributed computation for {@link org.jblas.DoubleMatrix}
 * @author Adam Gibson
 *
 */
public abstract class ComputableMasterImpl implements ComputableMaster<UpdateableSingleImpl>{

	protected UpdateableSingleImpl masterMatrix;
	
	@Override
	public void complete(DataOutputStream ds) {
		masterMatrix.get().write(ds);
	}

	@Override
	public UpdateableSingleImpl getResults() {
		return masterMatrix;
	}

	

}
