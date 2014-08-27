package org.deeplearning4j.datasets.mnist.draw;

import java.io.FileInputStream;
import java.io.ObjectInputStream;

import org.apache.commons.math3.random.MersenneTwister;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.dataset.DataSet;
import org.deeplearning4j.linalg.sampling.Sampling;
import org.deeplearning4j.nn.BaseNeuralNetwork;



public class LoadAndDraw {

	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {
		MnistDataSetIterator iter = new MnistDataSetIterator(60,60000);
		@SuppressWarnings("unchecked")
		ObjectInputStream ois = new ObjectInputStream(new FileInputStream(args[0]));
		
		BaseNeuralNetwork network = (BaseNeuralNetwork) ois.readObject();
		
		
		DataSet test = null;
		while(iter.hasNext()) {
			INDArray reconstructed = network.transform(test.getFeatureMatrix());
			for(int i = 0; i < test.numExamples(); i++) {
				INDArray draw1 = test.get(i).getFeatureMatrix().mul(255);
				INDArray reconstructed2 = reconstructed.getRow(i);
				INDArray draw2 = Sampling.binomial(reconstructed2, 1, new MersenneTwister(123)).mul(255);

				DrawReconstruction d = new DrawReconstruction(draw1);
				d.title = "REAL";
				d.draw();
				DrawReconstruction d2 = new DrawReconstruction(draw2,100,100);
				d2.title = "TEST";
				d2.draw();
				Thread.sleep(10000);
				d.frame.dispose();
				d2.frame.dispose();
			}
		}
		
		
	}

}
