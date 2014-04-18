package org.deeplearning4j.example.mnist;

import org.apache.commons.math3.random.MersenneTwister;
import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.RawMnistDataSetIterator;
import org.deeplearning4j.datasets.mnist.draw.DrawMnistGreyScale;
import org.deeplearning4j.dbn.GaussianRectifiedLinearDBN;
import org.deeplearning4j.distributions.Distributions;
import org.deeplearning4j.plot.FilterRenderer;
import org.deeplearning4j.rbm.CRBM;
import org.deeplearning4j.rbm.GaussianRectifiedLinearRBM;
import org.deeplearning4j.rbm.RBM;
import org.deeplearning4j.util.MatrixUtil;
import org.jblas.DoubleMatrix;
import org.slf4j.LoggerFactory;
import org.slf4j.Logger;

public class RawMnistRBMExample {

    private static Logger log = LoggerFactory.getLogger(RawMnistRBMExample.class);

	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {
		GaussianRectifiedLinearRBM r = new GaussianRectifiedLinearRBM.Builder()
		.numberOfVisible(784).useAdaGrad(true).withMomentum(0.3)
		.numHidden(600).normalizeByInputRows(true).useRegularization(false)
		.build();


		//batches of 10, 60000 examples total
		DataSetIterator iter = new RawMnistDataSetIterator(10,20);

		while(iter.hasNext()) {
			DataSet next = iter.next();
            next.scale();
            log.info("Data " + next);
			//train with k = 1 0.01 learning rate and 1000 epochs
			r.trainTillConvergence(next.getFirst(),1e-3, new Object[]{1,1e-3,1000});

		}



		iter.reset();






		//Iterate over the data set after done training and show the 2 side by side (you have to drag the test image over to the right)
		while(iter.hasNext()) {
			DataSet first = iter.next();
			DoubleMatrix reconstruct = r.reconstruct(first.getFirst());
			for(int j = 0; j < first.numExamples(); j++) {

				DoubleMatrix draw1 = first.get(j).getFirst().mul(255);
				DoubleMatrix reconstructed2 = reconstruct.getRow(j);
				DoubleMatrix draw2 = MatrixUtil.binomial(reconstructed2,1,new MersenneTwister(123)).mul(255);

				DrawMnistGreyScale d = new DrawMnistGreyScale(draw1);
				d.title = "REAL";
				d.draw();
				DrawMnistGreyScale d2 = new DrawMnistGreyScale(draw2,1000,1000);
				d2.title = "TEST";
				d2.draw();
				Thread.sleep(10000);
				d.frame.dispose();
				d2.frame.dispose();
			}


		}


	}

}
