package org.deeplearning4j.datasets.mnist.draw;

import org.apache.commons.math3.random.MersenneTwister;
import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.util.MatrixUtil;
import org.jblas.DoubleMatrix;


public class DrawMnist {
	public static void drawMnist(DataSet mnist,DoubleMatrix reconstruct) throws InterruptedException {
		for(int j = 0; j < mnist.numExamples(); j++) {
			DoubleMatrix draw1 = mnist.get(j).getFirst().mul(255);
			DoubleMatrix reconstructed2 = reconstruct.getRow(j);
			DoubleMatrix draw2 = MatrixUtil.binomial(reconstructed2,1,new MersenneTwister(123)).mul(255);

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
