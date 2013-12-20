package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda.matrix.jblas.iterativereduce.actor;

import com.ccc.sendalyzeit.textanalytics.ml.scaleout.iterativereduce.jblas.UpdateableMatrix;

public interface EpochDoneListener {

	void epochComplete(UpdateableMatrix result);
}
