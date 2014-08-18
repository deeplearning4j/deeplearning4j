package org.deeplearning4j.example.display;

import java.io.File;

import org.apache.commons.math3.random.MersenneTwister;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.LFWDataSetIterator;
import org.deeplearning4j.datasets.mnist.draw.DrawReconstruction;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.dataset.DataSet;
import org.deeplearning4j.linalg.sampling.Sampling;
import org.deeplearning4j.nn.BaseMultiLayerNetwork;
import org.deeplearning4j.plot.NeuralNetPlotter;
import org.deeplearning4j.scaleout.conf.Conf;
import org.deeplearning4j.util.SerializationUtils;

public class DisplayFiltersDBN {

	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {

		//batches of 10, 60000 examples total
		DataSetIterator iter = new LFWDataSetIterator(10,10);

		BaseMultiLayerNetwork network = SerializationUtils.readObject(new File(args[0]));
		
		NeuralNetPlotter plotter = new NeuralNetPlotter();
		
		//Iterate over the data applyTransformToDestination after done training and show the 2 side by side (you have to drag the test image over to the right)
		while(iter.hasNext()) {
			DataSet first = iter.next();
			
			plotter.plotNetworkGradient(network.getLayers()[0], network.getLayers()[0].getGradient(Conf.getDefaultRbmParams()),10);

			
			INDArray reconstruct = network.reconstruct(first.getFeatureMatrix(),0);
			for(int j = 0; j < first.numExamples(); j++) {

				INDArray draw1 = first.get(j).getFeatureMatrix().mul(255);
				INDArray reconstructed2 = reconstruct.getRow(j);
				INDArray draw2 = Sampling.binomial(reconstructed2, 1, new MersenneTwister(123)).mul(255);

				DrawReconstruction d = new DrawReconstruction(draw1);
				d.title = "REAL";
				d.draw();
				DrawReconstruction d2 = new DrawReconstruction(draw2,1000,1000);
				d2.title = "TEST";
				d2.draw();
				Thread.sleep(10000);
				d.frame.dispose();
				d2.frame.dispose();
			}


		}

	}

}
