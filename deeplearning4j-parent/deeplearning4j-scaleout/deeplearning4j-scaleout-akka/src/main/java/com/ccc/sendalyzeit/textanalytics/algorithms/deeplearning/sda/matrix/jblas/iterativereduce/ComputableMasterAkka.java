package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda.matrix.jblas.iterativereduce;

import java.util.Collection;

import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;

import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.nn.matrix.jblas.BaseMultiLayerNetwork;
import com.ccc.sendalyzeit.textanalytics.ml.scaleout.conf.Conf;
import com.ccc.sendalyzeit.textanalytics.ml.scaleout.iterativereduce.jblas.ComputableMasterMatrix;
import com.ccc.sendalyzeit.textanalytics.ml.scaleout.iterativereduce.jblas.UpdateableMatrix;

public class ComputableMasterAkka extends ComputableMasterMatrix implements DeepLearningConfigurable {

	


	@Override
	public UpdateableMatrix compute(Collection<UpdateableMatrix> workerUpdates,
			Collection<UpdateableMatrix> masterUpdates) {
		

		DeepLearningAccumulator acc = new DeepLearningAccumulator();
		for(UpdateableMatrix m : workerUpdates) 
			acc.accumulate(m.get());
		
		masterMatrix.set(acc.averaged());

		return masterMatrix;
	}

	@Override
	public void setup(Conf conf) {
		RandomGenerator rng =  new MersenneTwister(conf.getLong(SEED));
		BaseMultiLayerNetwork matrix = new BaseMultiLayerNetwork.Builder<>()
		.numberOfInputs(conf.getInt(N_IN)).numberOfOutPuts(conf.getInt(OUT)).withClazz(conf.getClazz(CLASS))
		.hiddenLayerSizes(conf.getIntsWithSeparator(LAYER_SIZES, ",")).withRng(rng)
		.build();
		masterMatrix = new UpdateableMatrix(matrix);


	}


}
