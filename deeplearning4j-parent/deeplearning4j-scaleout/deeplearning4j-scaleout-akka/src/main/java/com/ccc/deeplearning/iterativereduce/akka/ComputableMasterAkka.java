package com.ccc.deeplearning.iterativereduce.akka;

import java.util.Collection;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;

import com.ccc.deeplearning.nn.BaseMultiLayerNetwork;
import com.ccc.deeplearning.scaleout.conf.Conf;
import com.ccc.deeplearning.scaleout.conf.DeepLearningConfigurable;
import com.ccc.deeplearning.scaleout.iterativereduce.multi.ComputableMasterImpl;
import com.ccc.deeplearning.scaleout.iterativereduce.multi.UpdateableImpl;


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
		RandomGenerator rng =  new MersenneTwister(conf.getSeed());
		BaseMultiLayerNetwork matrix = new BaseMultiLayerNetwork.Builder<>()
		.numberOfInputs(conf.getnIn()).numberOfOutPuts(conf.getnOut()).withClazz(conf.getMultiLayerClazz())
		.hiddenLayerSizes(conf.getLayerSizes()).withRng(rng)
		.build();
		masterMatrix = new UpdateableImpl(matrix);


	}


}
