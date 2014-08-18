package org.deeplearning4j.example.lfw;

import java.io.BufferedOutputStream;
import java.io.FileOutputStream;
import java.util.Collections;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.SamplingDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.LFWDataSetIterator;
import org.deeplearning4j.dbn.DBN;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.linalg.api.activation.Activations;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.dataset.DataSet;
import org.deeplearning4j.rbm.RBM;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class LFWExample {

	private static Logger log = LoggerFactory.getLogger(LFWExample.class);
	
	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {
		//batches of 10, 60000 examples total
		DataSetIterator iter = new LFWDataSetIterator(10,100,28,28);
		DataSet load = iter.next(10);
        load.filterAndStrip(new int[]{1,2});
        load.normalizeZeroMeanZeroUnitVariance();
        load.sortByLabel();
        log.info("Data applyTransformToDestination " + load.numExamples());
        iter = new SamplingDataSetIterator(load,6,16);
		//784 input (number of columns in mnist, 10 labels (0-9), no regularization
        DBN dbn = new DBN.Builder()
         .withHiddenUnits(RBM.HiddenUnit.RECTIFIED)
         .withVisibleUnits(RBM.VisibleUnit.GAUSSIAN).lineSearchBackProp(true)
		.hiddenLayerSizes(new int[]{600, 500, 400}).useRegularization(true)
          .sampleFromHiddenActivations(true).withL2(1e-4).withSparsity(1e-1)
                .renderByLayer(Collections.singletonMap(0,10))
        .withDropOut(0.5).withActivation(Activations.tanh()).withMomentum(0.5)
		.numberOfInputs(iter.inputColumns()).numberOfOutPuts(load.numOutcomes())
		.build();

        while(iter.hasNext()) {
            DataSet next = iter.next();
            dbn.pretrain(next.getFeatureMatrix(), new Object[]{1, 1e-2, 1000, 1});

        }



		
		iter.reset();


		while(iter.hasNext()) {
			DataSet next = iter.next();
			dbn.setInput(next.getFeatureMatrix());
			dbn.finetune(next.getLabels(), 1e-3, 100);
		}
		
		
		BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream("lfw-dbn.bin"));
		dbn.write(bos);
		bos.flush();
		bos.close();
		log.info("Saved dbn");
		
		
		iter.reset();
		
		Evaluation eval = new Evaluation();
		
		while(iter.hasNext()) {
			DataSet next = iter.next();
			INDArray predict = dbn.output(next.getFeatureMatrix());
            INDArray labels = next.getLabels();
			eval.eval(labels, predict);
		}
		
		log.info("Prediction f scores and accuracy");
		log.info(eval.stats());
	}

}
