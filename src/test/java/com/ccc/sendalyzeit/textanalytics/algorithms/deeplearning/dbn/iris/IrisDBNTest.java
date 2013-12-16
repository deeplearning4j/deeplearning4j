package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.dbn.iris;

import static org.junit.Assert.*;

import java.io.IOException;

import org.apache.commons.math3.random.MersenneTwister;
import org.jblas.DoubleMatrix;
import org.jblas.SimpleBlas;
import org.junit.Test;

import com.ccc.sendalyzeit.deeplearning.berkeley.Pair;
import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.base.DeepLearningTest;
import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.dbn.matrix.jblas.CDBN;

public class IrisDBNTest extends DeepLearningTest {
	@Test
	public void testIris() throws IOException {
		Pair<DoubleMatrix,DoubleMatrix> data =getIris();
		int nIns = data.getFirst().columns;
		int nOuts = data.getSecond().columns;
		
		double preTrainLr = 0.1;
		int preTrainEpochs = 500;
		int k = 1;
		int[] hiddenLayerSizes = new int[] {300,300,300};
		double fineTuneLr = 0.1;
		int fineTuneEpochs = 300;

	
		
		CDBN dbn = new CDBN.Builder().numberOfInputs(nIns)
				.numberOfOutPuts(nOuts).withRng(new MersenneTwister(123))
				.hiddenLayerSizes(hiddenLayerSizes).build();
		dbn.pretrain(data.getFirst(),k, preTrainLr, preTrainEpochs);
		dbn.finetune(data.getSecond(), fineTuneLr, fineTuneEpochs);
		
		DoubleMatrix predict = dbn.predict(data.getFirst());
		
		for(int i = 0; i < data.getFirst().rows; i++) {
			DoubleMatrix trainingRow = data.getSecond().getRow(i);
			DoubleMatrix predictedRow = predict.getRow(i);
			
			int trainingMax = SimpleBlas.iamax(trainingRow);
			int predictedMax = SimpleBlas.iamax(predictedRow);
			assertEquals(trainingMax,predictedMax);
		}
	}

}
