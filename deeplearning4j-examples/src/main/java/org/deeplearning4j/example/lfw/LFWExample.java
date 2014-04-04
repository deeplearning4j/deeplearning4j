package org.deeplearning4j.example.lfw;

import java.io.BufferedOutputStream;
import java.io.FileOutputStream;

import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.LFWDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.RawMnistDataSetIterator;
import org.deeplearning4j.dbn.CDBN;
import org.deeplearning4j.dbn.GaussianRectifiedLinearDBN;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.activation.Activations;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class LFWExample {

	private static Logger log = LoggerFactory.getLogger(LFWExample.class);
	
	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {
		//batches of 10, 60000 examples total
		DataSetIterator iter = new LFWDataSetIterator(10,10);
		
		//784 input (number of columns in mnist, 10 labels (0-9), no regularization
		GaussianRectifiedLinearDBN dbn = new GaussianRectifiedLinearDBN.Builder().useAdaGrad(true).useRegularization(false)
		.hiddenLayerSizes(new int[]{500,400,250}).normalizeByInputRows(true)
		.numberOfInputs(iter.inputColumns()).numberOfOutPuts(iter.totalOutcomes())
		.build();
		
		while(iter.hasNext()) {
			DataSet next = iter.next();
			dbn.pretrain(next.getFirst(), 1, 0.001, 10000);
		}
		
		iter.reset();
		while(iter.hasNext()) {
			DataSet next = iter.next();
			dbn.setInput(next.getFirst());
			dbn.finetune(next.getSecond(), 0.001, 10000);
		}
		
		
		BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream("mnist-dbn.bin"));
		dbn.write(bos);
		bos.flush();
		bos.close();
		log.info("Saved dbn");
		
		
		iter.reset();
		
		Evaluation eval = new Evaluation();
		
		while(iter.hasNext()) {
			DataSet next = iter.next();
			DoubleMatrix predict = dbn.predict(next.getFirst());
			DoubleMatrix labels = next.getSecond();
			eval.eval(labels, predict);
		}
		
		log.info("Prediciton f scores and accuracy");
		log.info(eval.stats());
	}

}
