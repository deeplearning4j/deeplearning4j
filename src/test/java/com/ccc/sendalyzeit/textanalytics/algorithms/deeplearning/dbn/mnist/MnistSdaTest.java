package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.dbn.mnist;

import java.io.IOException;
import java.util.Arrays;

import org.apache.commons.math3.random.MersenneTwister;
import org.jblas.DoubleMatrix;
import org.junit.Test;

import com.ccc.sendalyzeit.deeplearning.berkeley.Pair;
import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.base.DeepLearningTest;
import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda.matrix.jblas.SdAMatrix;

public class MnistSdaTest extends DeepLearningTest {

	@Test
	public void testMnist() throws IOException {
		Pair<DoubleMatrix,DoubleMatrix> first = this.getMnistExample(1);
		int numIns = first.getFirst().columns;
		int numLabels = 10;
		int[] layerSizes = new int[4];
		int numEpochs = 1000;
		Arrays.fill(layerSizes,300);
		double lr = 0.1;
		SdAMatrix sda = new SdAMatrix.Builder().numberOfInputs(numIns)
				.numberOfOutPuts(numLabels).withRng(new MersenneTwister(123))
				.hiddenLayerSizes(layerSizes).build();
		
		for(int i = 2; i < 10; i++) {
			Pair<DoubleMatrix,DoubleMatrix> curr = this.getMnistExample(i);
			sda.pretrain(curr.getFirst(), lr, 0.3, numEpochs);
			sda.finetune(curr.getSecond(), lr, numEpochs);
		}
		
		
	
	}

}
