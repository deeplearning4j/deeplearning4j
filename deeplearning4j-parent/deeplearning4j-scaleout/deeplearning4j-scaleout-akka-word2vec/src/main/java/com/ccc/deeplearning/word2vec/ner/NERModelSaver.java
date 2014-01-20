package com.ccc.deeplearning.word2vec.ner;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import com.ccc.deeplearning.word2vec.Word2Vec;


public class NERModelSaver {

	private static Logger log = LoggerFactory.getLogger(NERModelSaver.class);


	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {
		String label = args[0];
		String trainingResource = args[1];

		NERClassifier classifier = null;
		//train and save
		if(args.length <= 4 ) {
			classifier = new NERClassifier(Arrays.asList(label));




			String type = label;
			@SuppressWarnings("unchecked")
			List<String> lines = IOUtils.readLines(new BufferedInputStream(new FileInputStream(new ClassPathResource(trainingResource).getFile())));
			for(int i = 0; i < lines.size(); i++) {
				String line = lines.get(i);
				if(line.isEmpty())
					continue;
				line = new InputHomogenization(line,Arrays.asList("<",">")).transform();
				line = line.replaceAll("<" + type.toLowerCase() + ">","<" + type + ">");	
				line = line.replaceAll("</" + type.toLowerCase() + ">","</" + type + ">");
				classifier.addExample(line);
			}

			

			classifier.train();
			classifier.save(new File(args[3]));


		}

		else {
			String path = args[3];
			BufferedInputStream bis = new BufferedInputStream(new FileInputStream(new File(path)));
			classifier = NERClassifier.load(bis);

		}


		BufferedReader br = 
				new BufferedReader(new InputStreamReader(System.in));
		String line = null;
		Word2Vec vec = classifier.getVec();
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
			List<String> write = classifier.predict(line);
			log.info("OUTPUT " + Arrays.asList(write));
		}

	}

}