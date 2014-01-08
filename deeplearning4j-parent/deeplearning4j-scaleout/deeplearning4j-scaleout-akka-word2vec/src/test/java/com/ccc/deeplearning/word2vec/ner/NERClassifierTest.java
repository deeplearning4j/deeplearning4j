package com.ccc.deeplearning.word2vec.ner;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.jblas.DoubleMatrix;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import com.ccc.deeplearning.word2vec.Word2Vec;
import com.ccc.deeplearning.word2vec.loader.Word2VecLoader;


public class NERClassifierTest {
	private static Logger log = LoggerFactory.getLogger(NERClassifier.class);
	@Test
	public void testNERClassifier() throws IOException {
		Word2Vec vec = Word2VecLoader.loadBinary(new ClassPathResource("/deeplearning/addressdeep").getFile());
		
		NERClassifier classifier = new NERClassifier(Arrays.asList("ADDRESS"));

		String type = "ADDRESS";
		List<String> lines = IOUtils.readLines(new BufferedInputStream(new FileInputStream(new ClassPathResource("/deeplearning/ADDRESS.split").getFile())));
		for(int i = 0; i < 1000; i++) {
			String line = lines.get(i);
			if(line.isEmpty())
				continue;
			line = new InputHomogenization(line,Arrays.asList("<",">")).transform();
			line = line.replaceAll("<" + type.toLowerCase() + ">","<" + type + ">");	
			line = line.replaceAll("</" + type.toLowerCase() + ">","</" + type + ">");
			classifier.addExample(line);
		}
	
		//assertEquals(true,Arrays.equals(new double[]{0,1.0},classifier.outcome("ADDRESS")));
		//double[][] inputs = classifier.getInputs();
		StringBuffer save = new StringBuffer();
		double[][] outputs = classifier.getOutputs();
		save.append("OUTPUTS " + Arrays.deepToString(outputs) + "\n");
		FileUtils.writeStringToFile(new File("saveresults"), save.toString());

		classifier.train();

		DoubleMatrix bs = classifier.classify("some random BS");
		DoubleMatrix address = classifier.classify("We live at 883 7th street Ántissa mAm Armado Equatorial Guinea erlabrunn 87534 in town.");
		log.info("BS " + bs.toString());
		log.info("ADDRESS " + address.toString());
		BufferedReader br = 
				new BufferedReader(new InputStreamReader(System.in));
		String line = null;
		while(true) {
			log.info("ENTER QUERY");
			line = br.readLine();
			if(line == null || line.equals("quit")) {
				log.info("QUIT");
				break;
			}
			else if(line.equals("ls")) {
				for(String key : vec.getVocab().keySet())
					log.info(key);
				continue;
			}
			else if(line.equals("count")) {
				log.info(String.valueOf(vec.getVocab().keySet().size()));
				continue;
			}
			line = new InputHomogenization(line,Arrays.asList("<",">")).transform();

			log.info("OUTPUT " + classifier.classify(line).toString());
		}
	}

}
