package com.ccc.deeplearning.sda.jblas.lfw;

import java.util.List;

import org.apache.commons.math3.random.MersenneTwister;
import org.jblas.DoubleMatrix;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.ccc.deeplearning.base.DeepLearningTest;
import com.ccc.deeplearning.berkeley.Pair;
import com.ccc.deeplearning.datasets.DataSet;
import com.ccc.deeplearning.datasets.iterator.impl.LFWDataSetIterator;
import com.ccc.deeplearning.eval.Evaluation;
import com.ccc.deeplearning.sda.StackedDenoisingAutoEncoder;
import com.ccc.deeplearning.util.MatrixUtil;

public class SDALFWTest extends DeepLearningTest {
   private static Logger log = LoggerFactory.getLogger(SDALFWTest.class);
   
	
	@Test
	public void testLfW() throws Exception {
		List<Pair<DoubleMatrix,DoubleMatrix>> lfw = getFirstFaces(1);
		LFWDataSetIterator iter = new LFWDataSetIterator(10,100);
		int numColumns = lfw.get(0).getFirst().columns;
		int outcomes = lfw.get(0).getSecond().columns;
		int[] layerSizes = new int[]{1000,1000,1000};
		
		StackedDenoisingAutoEncoder sda = new StackedDenoisingAutoEncoder.Builder()
		.hiddenLayerSizes(layerSizes).numberOfInputs(numColumns).useRegularization(false)
		.numberOfOutPuts(outcomes).withRng(new MersenneTwister(123)).build();

		
		while(iter.hasNext()) {
			DataSet next = iter.next();
			DoubleMatrix normalized = MatrixUtil.normalizeByColumnSums(next.getFirst());
			sda.pretrain(normalized, 0.1, 0.6, 2000);
			sda.finetune(next.getSecond(), 0.1, 500);
		}
		
		
		iter.reset();
		
		Evaluation eval = new Evaluation();
		
		while(iter.hasNext()) {
			DataSet next = iter.next();
			DoubleMatrix normalized = MatrixUtil.normalizeByColumnSums(next.getFirst());

			eval.eval(next.getSecond(), sda.predict(normalized));
		}
		
		log.info(eval.stats());
		
	}

}
