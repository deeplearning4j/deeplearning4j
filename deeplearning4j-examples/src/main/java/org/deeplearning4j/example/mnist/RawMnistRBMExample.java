package org.deeplearning4j.example.mnist;

import org.apache.commons.math3.random.MersenneTwister;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.RawMnistDataSetIterator;
import org.deeplearning4j.datasets.mnist.draw.DrawReconstruction;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.dataset.DataSet;
import org.deeplearning4j.linalg.sampling.Sampling;
import org.deeplearning4j.rbm.RBM;

import org.slf4j.LoggerFactory;
import org.slf4j.Logger;

public class RawMnistRBMExample {

    private static Logger log = LoggerFactory.getLogger(RawMnistRBMExample.class);

	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {
		RBM r = new RBM.Builder().withHidden(RBM.HiddenUnit.RECTIFIED).withVisible(RBM.VisibleUnit.GAUSSIAN)
		.numberOfVisible(784).useAdaGrad(true).withMomentum(0.3)
		.numHidden(600).useRegularization(false)
		.build();


		//batches of 10, 60000 examples total
		DataSetIterator iter = new RawMnistDataSetIterator(10,20);

		while(iter.hasNext()) {
			DataSet next = iter.next();
            next.scale();
            log.info("Data " + next);
			//train with k = 1 0.01 learning rate and 1000 epochs
			r.trainTillConvergence(next.getFeatureMatrix(),1e-3, new Object[]{1,1e-3,1000});

		}



		iter.reset();






		//Iterate over the data applyTransformToDestination after done training and show the 2 side by side (you have to drag the test image over to the right)
		while(iter.hasNext()) {
			DataSet first = iter.next();
			INDArray reconstruct = r.reconstruct(first.getFeatureMatrix());
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
