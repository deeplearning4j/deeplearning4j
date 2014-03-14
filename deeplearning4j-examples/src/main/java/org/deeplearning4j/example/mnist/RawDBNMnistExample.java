package org.deeplearning4j.example.mnist;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.util.List;
import java.util.concurrent.TimeUnit;

import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.RawMnistDataSetIterator;
import org.deeplearning4j.dbn.DBN;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.util.SerializationUtils;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class RawDBNMnistExample {

	private static Logger log = LoggerFactory.getLogger(RawDBNMnistExample.class);
	
	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {
		//batches of 10, 60000 examples total
		DataSetIterator iter = null;
		if(args.length < 2) {
			iter = new RawMnistDataSetIterator(10,60000);
		}
		else {
			int start = Integer.parseInt(args[1]);
			iter = new RawMnistDataSetIterator(60000,60000);
			DataSet next = iter.next();
			List<DataSet> list = next.asList();
			list = list.subList(start, list.size());
			iter = new ListDataSetIterator(list,10);
		}
		
		//784 input (number of columns in mnist, 10 labels (0-9), no regularization
		DBN dbn = null;
		
		if(args.length < 2) {
			dbn = new DBN.Builder()
			.hiddenLayerSizes(new int[]{500,400,250})
			.numberOfInputs(784).numberOfOutPuts(10)
			.useRegularization(false)
			.build();
			
		}
	
		else {
		     dbn = SerializationUtils.readObject(new File(args[0]));
		}
		
		int numIters = 0;
		
		while(iter.hasNext()) {
			DataSet next = iter.next();
			long now = System.currentTimeMillis();
			dbn.pretrain(next.getFirst(), 1, 0.01, 1000);
			long after = System.currentTimeMillis();
			log.info("Pretrain took " + TimeUnit.MILLISECONDS.toSeconds((after - now)) + " seconds");
			
			BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream("mnist-pretrain-dbn.bin-" + numIters));
			dbn.write(bos);
			bos.flush();
			bos.close();
			log.info("Saved dbn");
			numIters++;
			
			
			//dbn.finetune(next.getSecond(), 0.01, 1000);
		}
		
		
		BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream("mnist-dbn.bin"));
		dbn.write(bos);
		bos.flush();
		bos.close();
		log.info("Saved dbn");
		
		
		iter.reset();
		
		while(iter.hasNext()) {
			DataSet next = iter.next();
			dbn.finetune(next.getSecond(), 0.01, 1000);
		}
		
		iter.reset();

		
		Evaluation eval = new Evaluation();
		
		while(iter.hasNext()) {
			DataSet next = iter.next();
			DoubleMatrix predict = dbn.predict(next.getFirst());
			DoubleMatrix labels = next.getSecond();
			eval.eval(labels, predict);
		}
		
		log.info("Prediction f scores and accuracy");
		log.info(eval.stats());
		
	}

}
