package org.deeplearning4j.instrumentation;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.rbm.RBM;
import org.deeplearning4j.scaleout.conf.Conf;
import org.jblas.DoubleMatrix;
import org.junit.Test;

public class TestGradient {

	@Test
	public void testGetGradientAspect()  throws Exception {
		DataSetIterator mnist = new MnistDataSetIterator(10,10);
		
		RBM r = new RBM.Builder().numberOfVisible(784)
				.numHidden(5).build();
		
		r.setInput(mnist.next().getFirst());
		
		r.getGradient(Conf.getDefaultRbmParams());
	}

}
