package org.deeplearning4j.example.lfw;

import java.io.File;

import org.apache.commons.math3.random.MersenneTwister;
import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.LFWDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.RawMnistDataSetIterator;
import org.deeplearning4j.datasets.mnist.draw.DrawMnistGreyScale;
import org.deeplearning4j.distributions.Distributions;
import org.deeplearning4j.nn.NeuralNetwork.LossFunction;
import org.deeplearning4j.nn.NeuralNetwork.OptimizationAlgorithm;
import org.deeplearning4j.plot.FilterRenderer;
import org.deeplearning4j.rbm.CRBM;
import org.deeplearning4j.rbm.GaussianRectifiedLinearRBM;
import org.deeplearning4j.rbm.RBM;
import org.deeplearning4j.util.MatrixUtil;
import org.deeplearning4j.util.SerializationUtils;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class LFWRBMExample {

	private static Logger log = LoggerFactory.getLogger(LFWRBMExample.class);
	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {
		DataSetIterator iter = new LFWDataSetIterator(10,150000);
		int cols = iter.inputColumns();
		log.info("Learning from " + cols);

		GaussianRectifiedLinearRBM r = new GaussianRectifiedLinearRBM.Builder()
		.numberOfVisible(iter.inputColumns()).useAdaGrad(true).withOptmizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
		.numHidden(600).normalizeByInputRows(true).withMomentum(0.1).withDropOut(1).withLossFunction(LossFunction.RECONSTRUCTION_CROSSENTROPY)
		.build();

		for(int i = 0; i < 100; i++) {
			while(iter.hasNext()) {
				DataSet next = iter.next();
				next.divideBy(255);
				next.normalizeZeroMeanZeroUnitVariance();
				r.trainTillConvergence(next.getFirst(), 1e-2, new Object[]{1,1e-2,50});
				SerializationUtils.saveObject(r, new File("/home/agibsonccc/models/faces-rbm.bin"));


			}


			SerializationUtils.saveObject(r, new File("/home/agibsonccc/models/faces-rbm.bin"));
			iter.reset();

		}




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
