package org.deeplearning4j.gradient.multilayer;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.deeplearning4j.nn.gradient.MultiLayerGradient;
import org.jblas.DoubleMatrix;
/**
 * Collects a list of gradients
 * for averaging
 * @author Adam Gibson
 *
 */
public class AverageChangeMultiLayerGradientListener implements MultiLayerGradientListener {

	/**
	 * 
	 */
	private static final long serialVersionUID = 9078190492614228289L;
	private List<MultiLayerGradient> gradients = new ArrayList<>();
	
	
	@Override
	public void onMultiLayerGradient(MultiLayerGradient gradient) {
		gradients.add(gradient);
	}
	
	/**'
	 * Returns an averaged gradient
	 * @return an averaged gradient
	 */
	public MultiLayerGradient averaged() {
		MultiLayerGradient first = gradients.get(0).clone();
		for(int i = 1; i < gradients.size(); i++)
			first.addGradient(gradients.get(i).clone());
		first.div(gradients.size());
		return first;
	}
	
	
	


}
