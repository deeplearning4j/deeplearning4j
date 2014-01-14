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

import com.ccc.deeplearning.berkeley.Counter;
import com.ccc.deeplearning.berkeley.CounterMap;
import com.ccc.deeplearning.dbn.CDBN;
import com.ccc.deeplearning.util.MatrixUtil;
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

	/* initial statistics used for calculating word metadata such as word vectors to train on */
	private void calcWordFrequencies() throws IOException {
		for(File f : rootDir.listFiles())	 {

			for(File doc : f.listFiles()) {
				LineIterator iter = FileUtils.lineIterator(doc);
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
			}





		} 

	}


	public void train() throws IOException {

		calcWordFrequencies();

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

		for(File f : rootDir.listFiles())	 {

			for(File doc : f.listFiles()) {
				DoubleMatrix train = toWordCountVector(doc);
				cdbn.pretrain(train, 1, 0.1, 100);
				if(classify) {
					DoubleMatrix outcome = MatrixUtil.toOutcomeVector(labels.indexOf(f.getName()), labels.size());
					cdbn.finetune(outcome, 0.1, 100);
				}
			}
		}


	}

	/* Gather word count frequencies for a particular document */
	private DoubleMatrix toWordCountVector(File f) throws IOException {
		LineIterator iter = FileUtils.lineIterator(f);
		Counter<String> wordCounts = new Counter<String>();
		DoubleMatrix ret = new DoubleMatrix(vocab.size());
		while(iter.hasNext()) {
			String line = iter.nextLine();
			StringTokenizer tokenizer = new StringTokenizer(new InputHomogenization(line).transform());
			while(tokenizer.hasMoreTokens())  {
				String token = tokenizer.nextToken();
				if(!stopWords.contains(token)) {
					wordCounts.incrementCount(token,1.0);
				}

			}

		}

		for(String key : wordCounts.keySet()) {
			double count = wordCounts.getCount(key);
			int idx = vocab.indexOf(key);
			ret.put(idx,count);
		}

		ret.divi(vocab.size());

		return ret;
	}


	/* Compression/reconstruction */
	public DoubleMatrix reconstruct(String document) {
		return cdbn.reconstruct(toDocumentMatrix(document));
	}
	
	public DoubleMatrix labelDocument(String document) {
		DoubleMatrix d = toDocumentMatrix(document);
		return cdbn.predict(d);
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




	@SuppressWarnings("unchecked")
	private void readStopWords() {
		try {
			stopWords = IOUtils.readLines(new ClassPathResource("/stopwords").getInputStream());
		} catch (IOException e) {
			throw new RuntimeException(e);
		}

	}


}
