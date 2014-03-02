package org.deeplearning4j.iterativereduce.akka;

import java.util.Collection;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.nn.BaseMultiLayerNetwork;
import org.deeplearning4j.scaleout.conf.Conf;
import org.deeplearning4j.scaleout.conf.DeepLearningConfigurable;
import org.deeplearning4j.scaleout.iterativereduce.multi.ComputableMasterImpl;
import org.deeplearning4j.scaleout.iterativereduce.multi.UpdateableImpl;



public class ComputableMasterAkka extends ComputableMasterImpl implements DeepLearningConfigurable {

	


	@Override
	public UpdateableImpl compute(Collection<UpdateableImpl> workerUpdates,
			Collection<UpdateableImpl> masterUpdates) {
		

		DeepLearningAccumulator acc = new DeepLearningAccumulator();
		for(UpdateableImpl m : workerUpdates) 
			acc.accumulate(m.get());
		
		masterResults.set(acc.averaged());

		return masterResults;
	}

	@Override
	public void setup(Conf conf) {
		RandomGenerator rng =  new MersenneTwister(conf.getSeed());
		BaseMultiLayerNetwork matrix = new BaseMultiLayerNetwork.Builder<>()
		.numberOfInputs(conf.getnIn()).numberOfOutPuts(conf.getnOut()).withClazz(conf.getMultiLayerClazz())
		.hiddenLayerSizes(conf.getLayerSizes()).withRng(rng)
		.build();
		masterResults = new UpdateableImpl(matrix);


	}


}
