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
		MnistDataSetIterator fetcher = new MnistDataSetIterator(300,300);
		DataSet first = fetcher.next();
		int numIns = first.getFirst().columns;
		int numLabels = first.getSecond().columns;
		int[] layerSizes = {1000,500,250};
		double lr = 0.001;

		DBN dbn = new DBN.Builder().numberOfInputs(numIns)
				.renderWeights(2000).withMomentum(0).useRegularization(false)
				.numberOfOutPuts(numLabels).withRng(new MersenneTwister(123))
				.hiddenLayerSizes(layerSizes).build();
		
		do  {
			dbn.pretrain(first.getFirst(),1, lr, 2000);
			dbn.finetune(first.getSecond(),lr, 2000);
			/*RBM r = (RBM) dbn.layers[0];
			DoubleMatrix reconstruct = r.reconstruct(first.getFirst());
			
			for(int j = 0; j < first.numExamples(); j++) {
				DoubleMatrix draw1 = first.get(j).getFirst().mul(255);
				DoubleMatrix reconstructed2 = reconstruct.getRow(j);
				DoubleMatrix draw2 = MatrixUtil.binomial(reconstructed2,1,new MersenneTwister(123)).mul(255);

				DrawMnistGreyScale d = new DrawMnistGreyScale(draw1);
				d.title = "REAL";
				d.draw();
				DrawMnistGreyScale d2 = new DrawMnistGreyScale(draw2,100,100);
				d2.title = "TEST";
				d2.draw();
				Thread.sleep(1000);
				d.frame.dispose();
				d2.frame.dispose();

			}*/

			DoubleMatrix predicted = dbn.predict(first.getFirst());
			log.info("Predicting\n " + first.getSecond().toString().replaceAll(";","\n"));

			Evaluation eval = new Evaluation();
			eval.eval(first.getSecond(), predicted);
			log.info(eval.stats());
			if(fetcher.hasNext())
				first = fetcher.next();
		} while(fetcher.hasNext());

	}

}
