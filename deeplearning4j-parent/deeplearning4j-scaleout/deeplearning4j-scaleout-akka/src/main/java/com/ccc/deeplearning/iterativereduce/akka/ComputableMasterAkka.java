package com.ccc.deeplearning.iterativereduce.akka;

import java.util.Collection;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;

import com.ccc.deeplearning.nn.matrix.jblas.BaseMultiLayerNetwork;
import com.ccc.deeplearning.scaleout.conf.Conf;
import com.ccc.deeplearning.scaleout.conf.DeepLearningConfigurable;
import com.ccc.deeplearning.scaleout.iterativereduce.ComputableMasterImpl;
import com.ccc.deeplearning.scaleout.iterativereduce.UpdateableImpl;


public class ComputableMasterAkka extends ComputableMasterImpl implements DeepLearningConfigurable {

	


	@Override
	public UpdateableImpl compute(Collection<UpdateableImpl> workerUpdates,
			Collection<UpdateableImpl> masterUpdates) {
		

		DeepLearningAccumulator acc = new DeepLearningAccumulator();
		for(UpdateableImpl m : workerUpdates) 
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
		masterMatrix = new UpdateableImpl(matrix);


	}


}
