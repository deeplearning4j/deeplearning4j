package com.ccc.deeplearning.dbn.matrix.jblas.mnist;

import java.io.IOException;

import org.apache.commons.math3.random.MersenneTwister;
import org.jblas.DoubleMatrix;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.ccc.deeplearning.base.DeepLearningTest;
import com.ccc.deeplearning.datasets.DataSet;
import com.ccc.deeplearning.datasets.iterator.impl.MnistDataSetIterator;
import com.ccc.deeplearning.datasets.mnist.draw.DrawMnistGreyScale;
import com.ccc.deeplearning.dbn.DBN;
import com.ccc.deeplearning.eval.Evaluation;
import com.ccc.deeplearning.rbm.RBM;
import com.ccc.deeplearning.util.MatrixUtil;

public class MnistDbnTest extends DeepLearningTest {

	private static Logger log = LoggerFactory.getLogger(MnistDbnTest.class);

	@Test
	public void testMnist() throws IOException, InterruptedException {
		MnistDataSetIterator fetcher = new MnistDataSetIterator(50,2000);
		DataSet first = fetcher.next();
		int numIns = first.getFirst().columns;
		int numLabels = first.getSecond().columns;
		int[] layerSizes = {500,500,500};
		double lr = 0.1;

		DBN dbn = new DBN.Builder().numberOfInputs(numIns)
				.renderWeights(0).withMomentum(0.9).useRegularization(false)
				.numberOfOutPuts(numLabels).withRng(new MersenneTwister(123))
				.hiddenLayerSizes(layerSizes).build();

		do  {
			dbn.pretrain(first.getFirst(),1, lr, 300);

			if(fetcher.hasNext())
				first = fetcher.next();
		} while(fetcher.hasNext());

		fetcher.reset();
		first = fetcher.next();
		
		do {
			dbn.finetune(first.getSecond(),lr, 300);
			
			if(fetcher.hasNext())
				first = fetcher.next();
		}while(fetcher.hasNext());

		fetcher.reset();
		first = fetcher.next();
		Evaluation eval = new Evaluation();

		do {


			DoubleMatrix predicted = dbn.predict(first.getFirst());
			log.info("Predicting\n " + first.getSecond().toString().replaceAll(";","\n"));
			log.info("Prediction was " + predicted.toString().replaceAll(";","\n"));
			eval.eval(first.getSecond(), predicted);
			if(fetcher.hasNext())
				first = fetcher.next();
		}while(fetcher.hasNext());
		

		log.info(eval.stats());


	}

}
