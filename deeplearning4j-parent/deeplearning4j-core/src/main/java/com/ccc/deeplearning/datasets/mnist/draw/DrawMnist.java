package com.ccc.deeplearning.datasets.mnist.draw;

import org.apache.commons.math3.random.MersenneTwister;
import org.jblas.DoubleMatrix;

import com.ccc.deeplearning.datasets.DataSet;
import com.ccc.deeplearning.util.MatrixUtil;

public class DrawMnist {
	public static void drawMnist(DataSet mnist,DoubleMatrix reconstruct) throws InterruptedException {
		for(int j = 0; j < mnist.numExamples(); j++) {
			DoubleMatrix draw1 = mnist.get(j).getFirst().mul(255);
			DoubleMatrix reconstructed2 = reconstruct.getRow(j);
			DoubleMatrix draw2 = MatrixUtil.binomial(reconstructed2,1,new MersenneTwister(123)).mul(255);

			DrawMnistGreyScale d = new DrawMnistGreyScale(draw1);
			d.title = "REAL";
			d.draw();
			DrawMnistGreyScale d2 = new DrawMnistGreyScale(draw2,1000,1000);
			d2.title = "TEST";
			
			d2.draw();
			Thread.sleep(1000);
			d.frame.dispose();
			d2.frame.dispose();

		}
	}

}
