package org.deeplearning4j.gradient.multilayer;

import java.util.ArrayList;
import java.util.List;

import org.deeplearning4j.nn.gradient.MultiLayerGradient;
import org.deeplearning4j.plot.NeuralNetPlotter;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class WeightPlotListener implements MultiLayerGradientListener {

	/**
	 * 
	 */
	private static final long serialVersionUID = -2476819215506562426L;

	private List<MultiLayerGradient> gradients = new ArrayList<>();
	private static Logger log = LoggerFactory.getLogger(WeightPlotListener.class);
	
	
	@Override
	public void onMultiLayerGradient(MultiLayerGradient gradient) {
		
		
		gradients.add(gradient);
		
		if(gradients.size() >= 6)
			gradients.remove(0);
		
		
		plot();
		
	}
	
	
	public void plot() {
		
		DoubleMatrix[] d = new DoubleMatrix[gradients.size()];
		String[] names = new String[gradients.size()];
		log.info("Plotting " + gradients.size() + " matrices");
		for(int i = 0; i < gradients.size(); i++) {
			names[i] = String.valueOf(i);
			d[i] = gradients.get(i).getGradients().get(0).getwGradient();
		}
		
		NeuralNetPlotter plotter = new NeuralNetPlotter();
		plotter.plotMatrices(names, d);
		
		
	}

	

}
