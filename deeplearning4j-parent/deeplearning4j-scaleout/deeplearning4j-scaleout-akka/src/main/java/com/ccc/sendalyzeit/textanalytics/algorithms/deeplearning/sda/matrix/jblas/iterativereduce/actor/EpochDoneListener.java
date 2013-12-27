package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda.matrix.jblas.iterativereduce.actor;

import java.io.Serializable;

import com.ccc.sendalyzeit.textanalytics.ml.scaleout.iterativereduce.jblas.UpdateableMatrix;

public interface EpochDoneListener extends Serializable {

	void epochComplete(UpdateableMatrix result);
	
	void finish();
}
