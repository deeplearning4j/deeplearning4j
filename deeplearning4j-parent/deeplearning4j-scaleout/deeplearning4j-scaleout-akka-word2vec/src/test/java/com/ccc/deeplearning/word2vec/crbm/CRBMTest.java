package com.ccc.deeplearning.word2vec.crbm;

import java.io.File;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.apache.commons.io.FileUtils;
import org.apache.commons.math3.random.MersenneTwister;
import org.jblas.DoubleMatrix;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import com.ccc.deeplearning.rbm.matrix.jblas.CRBM;
import com.ccc.deeplearning.word2vec.Word2Vec;
import com.ccc.deeplearning.word2vec.util.Window;
import com.ccc.deeplearning.word2vec.util.WindowConverter;
import com.ccc.deeplearning.word2vec.util.Windows;

public class CRBMTest {

	private static Logger log = LoggerFactory.getLogger(CRBMTest.class);


	@SuppressWarnings("unchecked")
	@Test
	public void testTrain() throws Exception {
		ClassPathResource model = new ClassPathResource("/word2vecmodel.bin");
		File f = model.getFile();

		Word2Vec vec = new Word2Vec();
		vec.loadModel(f);

		int nIn = vec.getWindow() * vec.getSyn0().columns;
		CRBM r = null;

		File dir = new File("src/test/resources/articles");

		Iterator<File> files = FileUtils.iterateFiles(dir, null, true);
		List<DoubleMatrix> trainingExamples = new ArrayList<DoubleMatrix>();

		while(files.hasNext()) {
			File next = files.next();
			List<String> lines = FileUtils.readLines(next);

			for(String line : lines) {
				List<Window> windows = Windows.windows(line);
				for(Window window : windows) {
					DoubleMatrix input = new DoubleMatrix(WindowConverter.asExample(window, vec)).transpose();
					trainingExamples.add(input);


				}
			}
			DoubleMatrix input = new DoubleMatrix(trainingExamples.size(),trainingExamples.get(0).columns);
			for(int i = 0; i < trainingExamples.size(); i++) {
				DoubleMatrix example = trainingExamples.get(i);
				input.putRow(i,example);
			}

			if(r == null) {
				r = new CRBM.Builder()
				.numberOfVisible(input.columns)
				.numHidden(500)
				.withL2(0.1)
				.withMomentum(0.9)
				.withRandom(new MersenneTwister(123))
				.build();

			}

			log.info("Number of inputs " + nIn + " and number of columns " + trainingExamples.get(0).columns);
			for(int i = 0; i < 10; i++)
				r.trainTillConvergence(0.1, 1, input);


		}


		log.info("Final cross entropy " + r.getReConstructionCrossEntropy());



	}

}
