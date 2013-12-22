package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda.lfw;

import java.util.List;

import org.apache.commons.math3.random.MersenneTwister;
import org.jblas.DoubleMatrix;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.ccc.sendalyzeit.deeplearning.berkeley.Pair;
import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.base.DeepLearningTest;
import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda.matrix.jblas.SdAMatrix;

public class SDALFWTest extends DeepLearningTest {
   private static Logger log = LoggerFactory.getLogger(SDALFWTest.class);
   
	
	@Test
	public void testLfW() throws Exception {
		List<Pair<DoubleMatrix,DoubleMatrix>> lfw = this.getFirstFaces(100);
		int numColumns = lfw.get(0).getFirst().columns;
		int outcomes = lfw.get(0).getSecond().columns;
		int[] layerSizes = new int[]{200,200,200};
		SdAMatrix sda = new SdAMatrix.Builder().hiddenLayerSizes(layerSizes).numberOfInputs(numColumns).numberOfOutPuts(outcomes).withRng(new MersenneTwister(123)).build();
		for(int i = 0; i < 1; i++) {
			sda.pretrain(lfw.get(i).getFirst(), 0.1, 0.3, 200);
			sda.finetune(lfw.get(i).getSecond(), 0.1, 500);
		}
		
		log.info(sda.predict(lfw.get(0).getFirst()).toString());
		
	}

}
