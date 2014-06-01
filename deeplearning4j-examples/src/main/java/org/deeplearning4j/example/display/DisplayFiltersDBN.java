package org.deeplearning4j.example.display;

import java.io.File;

import org.apache.commons.math3.random.MersenneTwister;
import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.LFWDataSetIterator;
import org.deeplearning4j.datasets.mnist.draw.DrawReconstruction;
import org.deeplearning4j.nn.BaseMultiLayerNetwork;
import org.deeplearning4j.plot.NeuralNetPlotter;
import org.deeplearning4j.scaleout.conf.Conf;
import org.deeplearning4j.util.MatrixUtil;
import org.deeplearning4j.util.SerializationUtils;
import org.jblas.DoubleMatrix;

public class DisplayFiltersDBN {

	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {

		//batches of 10, 60000 examples total
		DataSetIterator iter = new LFWDataSetIterator(10,10);

		BaseMultiLayerNetwork network = SerializationUtils.readObject(new File(args[0]));
		
		NeuralNetPlotter plotter = new NeuralNetPlotter();
		
		//Iterate over the data set after done training and show the 2 side by side (you have to drag the test image over to the right)
		while(iter.hasNext()) {
			DataSet first = iter.next();
			
			plotter.plotNetworkGradient(network.getLayers()[0], network.getLayers()[0].getGradient(Conf.getDefaultRbmParams()),10);

			
			DoubleMatrix reconstruct = network.reconstruct(first.getFirst(),0);
			for(int j = 0; j < first.numExamples(); j++) {

				DoubleMatrix draw1 = first.get(j).getFirst().mul(255);
				DoubleMatrix reconstructed2 = reconstruct.getRow(j);
				DoubleMatrix draw2 = MatrixUtil.binomial(reconstructed2,1,new MersenneTwister(123)).mul(255);

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
