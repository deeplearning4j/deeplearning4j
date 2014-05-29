package org.deeplearning4j.datasets.mnist.draw;

import java.io.FileInputStream;
import java.io.ObjectInputStream;

import org.apache.commons.math3.random.MersenneTwister;
import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.BaseNeuralNetwork;
import org.deeplearning4j.util.MatrixUtil;
import org.jblas.DoubleMatrix;


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
			test = iter.next();
			DoubleMatrix reconstructed = network.reconstruct(test.getFirst());
			for(int i = 0; i < test.numExamples(); i++) {
				DoubleMatrix draw1 = test.get(i).getFirst().mul(255);
				DoubleMatrix reconstructed2 = reconstructed.getRow(i);
				DoubleMatrix draw2 = MatrixUtil.binomial(reconstructed2,1,new MersenneTwister(123)).mul(255);

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
