package com.ccc.deeplearning.topicmodeling;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.apache.commons.io.LineIterator;
import org.jblas.DoubleMatrix;
import org.springframework.core.io.ClassPathResource;

import com.ccc.deeplearning.berkeley.CounterMap;
import com.ccc.deeplearning.dbn.CDBN;
import com.ccc.deeplearning.word2vec.ner.InputHomogenization;
import com.ccc.deeplearning.word2vec.viterbi.Index;

/**
 * Topic Modeling
 * Using a CDBN as a deep auto encoder
 * This will collect word count frequencies from directories as follows:
 * The root directory contains several subdirectories each of a different label.
 * Word count frequencies against labels are then constructed using those sub directories.
 * 
 * From there, an input matrix is created that contains
 * normalized probabilities of the wordcount frequencies against the labels.
 * Depending on the numOuts parameter you can use this to visualize documents
 * or condense word count vectors in to more compact representations
 * that maybe used to train a classifier or do a search. 
 * @author Adam Gibson
 *
 */
public class TopicModeling {

	private List<String> labels;
	private CDBN cdbn;
	private CounterMap<String,String> words = new CounterMap<String,String>();
	private Index vocab = new Index();
	private List<String> stopWords;
	private File rootDir;
	private int numOuts;
	private boolean classify;

	/**
	 * Creates a topic modeler
	 * This can either be used for information compression or
	 * classification via reconstruction of input documents
	 * @param rootDir the root directory to train on
	 * @param numOuts the number of outputs for compression
	 * or input in to the subsequent layers for classification
	 * @param classify whether to classify or compress
	 */
	public TopicModeling(File rootDir,int numOuts,boolean classify) {
		this.numOuts = numOuts;
		File[] subs = rootDir.listFiles();

		labels = new ArrayList<String>(subs.length);
		for(File f : subs) {
			labels.add(f.getName());
		}
		readStopWords();

	}



	public void train() {
		for(File f : rootDir.listFiles())	 {
			try {
				LineIterator iter = FileUtils.lineIterator(f);
				while(iter.hasNext()) {
					String line = iter.nextLine();
					StringTokenizer tokenizer = new StringTokenizer(new InputHomogenization(line).transform());
					while(tokenizer.hasMoreTokens())  {
						String token = tokenizer.nextToken();
						if(!stopWords.contains(token)) {
							words.incrementCount(token,f.getName(), 1.0);
							if(vocab.indexOf(token) < 0)
								vocab.add(token);
						}

					}

				}



			} catch (IOException e) {
				throw new RuntimeException(e);
			}

		}

		//create a network that knows how to reconstruct the input documents
		if(classify)
			cdbn = new CDBN.Builder()
		.numberOfInputs(words.keySet().size())
		.numberOfOutPuts(labels.size())
		.hiddenLayerSizes(new int[]{words.keySet().size() * 2,words.keySet().size(),numOuts,words.keySet().size() * 2,words.keySet().size()})
		.build();

		//create a network that knows how to compress documents
		else
			cdbn = new CDBN.Builder()
		.numberOfInputs(words.keySet().size())
		.numberOfOutPuts(labels.size())
		.hiddenLayerSizes(new int[]{words.keySet().size() * 2,words.keySet().size(),numOuts})
		.build();


		DoubleMatrix input = toMatrix();
		cdbn.pretrain(input,1, 0.01, 1000);
		
		


	}


	public DoubleMatrix reconstruct(String document) {
		return cdbn.reconstruct(toDocumentMatrix(document));
	}

	private DoubleMatrix toDocumentMatrix(String input) {
		StringTokenizer tokenizer = new StringTokenizer(input);
		DoubleMatrix ret = new DoubleMatrix(vocab.size());
		while(tokenizer.hasMoreTokens())  {
			String token = tokenizer.nextToken();
			if(!stopWords.contains(token)) {
				int idx = vocab.indexOf(token);
				if(idx >= 0) {
					ret.put(idx,words.getCounter(token).totalCount());
				}
			}

		}
		//normalize the word counts by the number of words
		ret.divi(ret.length);
		return ret;
	}

	private DoubleMatrix toMatrix() {
		DoubleMatrix ret = new DoubleMatrix(vocab.size(), labels.size());
		for(int i = 0; i < vocab.size(); i++) {
			DoubleMatrix row = new DoubleMatrix(labels.size());
			String word = (String) vocab.get(i);
			for(int l = 0; l < row.length; l++) {
				double count = words.getCount(word, labels.get(l));
				row.put(l,count);
			}
		}

		ret.divi(ret.rowSums());

		return ret;
	}


	@SuppressWarnings("unchecked")
	private void readStopWords() {
		try {
			stopWords = IOUtils.readLines(new ClassPathResource("/stopwords").getInputStream());
		} catch (IOException e) {
			throw new RuntimeException(e);
		}

	}


}
