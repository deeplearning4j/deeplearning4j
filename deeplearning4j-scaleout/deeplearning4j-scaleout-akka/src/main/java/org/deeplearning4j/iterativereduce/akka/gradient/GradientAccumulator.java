package org.deeplearning4j.iterativereduce.akka.gradient;

import java.util.ArrayList;
import java.util.List;

import org.deeplearning4j.nn.gradient.MultiLayerGradient;
import org.deeplearning4j.nn.gradient.NeuralNetworkGradient;


/**
 * Accumulates and produces an average gradient
 * @author Adam Gibson
 *
 */
public class GradientAccumulator {

	private List<MultiLayerGradient> workers = new ArrayList<MultiLayerGradient>();

	public void accumulate(MultiLayerGradient sda) {
		workers.add(sda);
	}

	public MultiLayerGradient averaged() {
		if(workers.isEmpty())
			return null;
		MultiLayerGradient gradient  = workers.get(0);
		List<NeuralNetworkGradient> firstGradient = gradient.getGradients();
		for(int i = 1; i < workers.size(); i++) {
			List<NeuralNetworkGradient> gradients = workers.get(i).getGradients();
			for(int j = 0; j < gradients.size();j++) {
				firstGradient.get(j).add(gradients.get(j));
			}
			
			gradient.getLogRegGradient().add(workers.get(i).getLogRegGradient());
			
		}
	
		for(NeuralNetworkGradient g : gradient.getGradients()) {
			g.div(workers.size());
		}
		
		gradient.getLogRegGradient().div(workers.size());
		
		return gradient;
	}

}
