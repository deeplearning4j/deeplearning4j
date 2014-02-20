package com.ccc.deeplearning.word2vec.da;

import java.io.File;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.StringTokenizer;

import org.apache.commons.io.FileUtils;
import org.apache.commons.math3.random.MersenneTwister;
import org.deeplearning4j.da.DenoisingAutoEncoder;
import org.deeplearning4j.rbm.CRBM;
import org.deeplearning4j.util.MatrixUtil;
import org.deeplearning4j.word2vec.Word2Vec;
import org.deeplearning4j.word2vec.loader.Word2VecLoader;
import org.deeplearning4j.word2vec.ner.InputHomogenization;
import org.deeplearning4j.word2vec.util.Window;
import org.deeplearning4j.word2vec.util.WindowConverter;
import org.deeplearning4j.word2vec.util.Windows;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;


public class DATest {

	private static Logger log = LoggerFactory.getLogger(DATest.class);


	@SuppressWarnings("unchecked")
	@Test
	public void testTrain() throws Exception {
		ClassPathResource model = new ClassPathResource("/word2vec-address.bin");
		File f = model.getFile();

		Word2Vec vec = Word2VecLoader.loadModel(f);

		int nIn = vec.getWindow() * vec.getSyn0().columns;
		DenoisingAutoEncoder r = null;

		File dir = new File("src/test/resources/deeplearning");

		Iterator<File> files = FileUtils.iterateFiles(dir, null, true);
		List<DoubleMatrix> trainingExamples = new ArrayList<DoubleMatrix>();
		List<String> allLines = new ArrayList<String>();

		while(files.hasNext()) {
			File next = files.next();
			List<String> lines = FileUtils.readLines(next);

			for(String line : lines) {
				line = new InputHomogenization(line).transform();
				allLines.add(line);
				List<Window> windows = Windows.windows(line);
				for(Window window : windows) {
					DoubleMatrix input = new DoubleMatrix(WindowConverter.asExample(window, vec)).transpose();
					MatrixUtil.normalize(input);
					
					trainingExamples.add(input);


				}
			}






		}



		int numExamples = 10000;
		
		DoubleMatrix input = new DoubleMatrix(numExamples,trainingExamples.get(0).columns);
		for(int i = 0; i < numExamples; i++) {
			DoubleMatrix example = trainingExamples.get(i);
			input.putRow(i,example);
		}
		

		if(r == null) {
			r = new DenoisingAutoEncoder.Builder()
			.numberOfVisible(input.columns)
			.numHidden(500)
			.withL2(0.1)
			.withMomentum(0.9)
			.withRandom(new MersenneTwister(123))
			.build();

		}

		log.info("Number of inputs " + nIn + " and number of columns " + trainingExamples.get(0).columns);
		String example = allLines.get(0);
		double corruption = 0.3;
		List<Window> windows = Windows.windows(example);
		StringBuffer errors = new StringBuffer();
		
		
		for(int i = 0; i < 2000; i++) {


			r.trainTillConverge(input, 0.1,corruption);
			/*StringTokenizer tokenizer = new StringTokenizer(new InputHomogenization(example).transform());
			while(tokenizer.hasMoreTokens()) {
				String token = tokenizer.nextToken();
				if(vec.getVocab().containsKey(token)) {
					DoubleMatrix curr = vec.getWordVectorMatrix(token).mul(0.0001);
					DoubleMatrix gradient = curr.sub(vec.getWordVectorMatrix(token));
					DoubleMatrix add = vec.getWordVectorMatrix(token).add(gradient);
					int idx = vec.indexOf(token);
					vec.getSyn0().putRow(idx, add);
					log.info("Updated word " + token);
				}
			}

			input = new DoubleMatrix(windows.size(),trainingExamples.get(0).columns);
			for(int w = 0; w < windows.size(); w++) {
				input.putRow(w,new DoubleMatrix(WindowConverter.asExample(windows.get(w), vec)).transpose());
			}*/
			double entropy = r.negativeLoglikelihood(corruption);
			errors.append(i + "," + entropy + "\n");
			log.info("cross entropy " + entropy);

		}
		
	
		FileUtils.writeStringToFile(new File("/home/agibsonccc/Desktop/errors.csv"), errors.toString());
		
		log.info("End " +   r.negativeLoglikelihood(corruption));
		
		for(int i = 0; i < windows.size(); i++) {
			DoubleMatrix test = new DoubleMatrix(WindowConverter.asExample(windows.get(i), vec)).transpose();
			test = MatrixUtil.normalize(test);
			String testWords = windows.get(i).asTokens();
			DoubleMatrix reconstructed = MatrixUtil.normalize(r.reconstruct(test));
			log.info("Reconstruct " + testWords  + " was " + reconstructed + " with example " + test);
		}

	}

}
