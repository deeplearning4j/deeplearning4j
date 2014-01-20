package com.ccc.deeplearning.dbn.matrix.jblas.mnist;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.math3.random.MersenneTwister;
import org.jblas.DoubleMatrix;
import org.jblas.SimpleBlas;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.ccc.deeplearning.base.DeepLearningTest;
import com.ccc.deeplearning.berkeley.Pair;
import com.ccc.deeplearning.datasets.DataSet;
import com.ccc.deeplearning.datasets.fetchers.MnistDataFetcher;
import com.ccc.deeplearning.dbn.CDBN;
import com.ccc.deeplearning.dbn.DBN;
import com.ccc.deeplearning.eval.Evaluation;

public class MnistDbnTest extends DeepLearningTest {

	private static Logger log = LoggerFactory.getLogger(MnistDbnTest.class);

	@Test
	public void testMnist() throws IOException {
		MnistDataFetcher fetcher = new MnistDataFetcher();
		fetcher.fetch(8000);
		DataSet first = fetcher.next();
		int numIns = first.getFirst().columns;
		int numLabels = first.getSecond().columns;
		int[] layerSizes = {1000,1000,2000};
		double lr = 0.1;
		
		DBN dbn = new DBN.Builder().numberOfInputs(numIns)
				.numberOfOutPuts(numLabels).withRng(new MersenneTwister(123))
				.hiddenLayerSizes(layerSizes).build();
	
		dbn.pretrain(first.getFirst(),1, lr, 50);
		dbn.finetune(first.getSecond(),lr, 50);
		
		
		DoubleMatrix predicted = dbn.predict(first.getFirst());
		//log.info("Predicting\n " + first.getFirst().toString().replaceAll(";","\n"));

		Evaluation eval = new Evaluation();
		eval.eval(first.getSecond(), predicted);
		log.info(eval.stats());

	}

}
