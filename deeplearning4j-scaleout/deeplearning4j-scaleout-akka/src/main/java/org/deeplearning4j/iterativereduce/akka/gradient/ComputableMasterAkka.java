package org.deeplearning4j.iterativereduce.akka.gradient;

import java.util.Collection;

import org.deeplearning4j.scaleout.conf.Conf;
import org.deeplearning4j.scaleout.conf.DeepLearningConfigurable;
import org.deeplearning4j.scaleout.iterativereduce.multi.gradient.ComputableMasterImpl;
import org.deeplearning4j.scaleout.iterativereduce.multi.gradient.UpdateableGradientImpl;



public class ComputableMasterAkka extends ComputableMasterImpl implements DeepLearningConfigurable {

	


	@Override
	public UpdateableGradientImpl compute(Collection<UpdateableGradientImpl> workerUpdates,
			Collection<UpdateableGradientImpl> masterUpdates) {
		
		GradientAccumulator acc = new GradientAccumulator();
		
		for(UpdateableGradientImpl m : workerUpdates) 
			acc.accumulate(m.get());
		
		
		
		masterResults.set(acc.averaged());

		return masterResults;
	}

	@Override
	public void setup(Conf conf) {
		
	}




}
