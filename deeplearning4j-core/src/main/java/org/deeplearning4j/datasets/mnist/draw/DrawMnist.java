package org.deeplearning4j.datasets.mnist.draw;

import org.apache.commons.math3.random.MersenneTwister;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.dataset.DataSet;
import org.deeplearning4j.linalg.sampling.Sampling;


public class DrawMnist {
	public static void drawMnist(DataSet mnist,INDArray reconstruct) throws InterruptedException {
		for(int j = 0; j < mnist.numExamples(); j++) {
			INDArray draw1 = mnist.get(j).getFeatureMatrix().mul(255);
			INDArray reconstructed2 = reconstruct.getRow(j);
			INDArray draw2 = Sampling.binomial(reconstructed2, 1, new MersenneTwister(123)).mul(255);

			DrawReconstruction d = new DrawReconstruction(draw1);
			d.title = "REAL";
			d.draw();
			DrawReconstruction d2 = new DrawReconstruction(draw2,1000,1000);
			d2.title = "TEST";
			
			d2.draw();
			Thread.sleep(1000);
			d.frame.dispose();
			d2.frame.dispose();

		}
	}

}
