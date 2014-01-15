package com.ccc.deeplearning.topicmodeling;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.StringTokenizer;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.apache.commons.io.LineIterator;
import org.apache.commons.lang3.StringUtils;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import com.ccc.deeplearning.berkeley.Counter;
import com.ccc.deeplearning.berkeley.CounterMap;
import com.ccc.deeplearning.datasets.DataSet;
import com.ccc.deeplearning.dbn.CDBN;
import com.ccc.deeplearning.eval.Evaluation;
import com.ccc.deeplearning.nn.activation.HardTanh;
import com.ccc.deeplearning.util.MathUtils;
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
	private CDBN cdbn,discriminatory;
	private CounterMap<String,String> words = new CounterMap<String,String>();
	private Index vocab = new Index();
	private List<String> stopWords;
	private File rootDir;
	private int numOuts;
	private boolean classify;
	private DataSet trained;
	private static Logger log = LoggerFactory.getLogger(TopicModeling.class);

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
		this.rootDir = rootDir;
		this.classify = classify;
		File[] subs = rootDir.listFiles();

		labels = new ArrayList<String>(subs.length);
		for(File f : subs) {
			labels.add(f.getName());
		}
		readStopWords();

	}

	/* initial statistics used for calculating word metadata such as word vectors to train on */
	private void calcWordFrequencies() throws IOException {
		Counter<String> tf = new Counter<String>();
		Counter<String> idf = new Counter<String>();
		for(File f : rootDir.listFiles())	 {

			for(File doc : f.listFiles()) {
				Set<String> encountered = new HashSet<String>();
				LineIterator iter = FileUtils.lineIterator(doc);
				while(iter.hasNext()) {
					String line = iter.nextLine();
					StringTokenizer tokenizer = new StringTokenizer(new InputHomogenization(line).transform());
					while(tokenizer.hasMoreTokens())  {
						String token = tokenizer.nextToken();
						if(!stopWords.contains(token)) {
							words.incrementCount(token,f.getName(), 1.0);
							tf.incrementCount(token, 1.0);
							if(!encountered.contains(token)) {
								idf.incrementCount(token, 1.0);
								encountered.add(token);
							}
						}

					}

				}			
			}
		} 


		Counter<String> tfidf = new Counter<String>();
		//term frequency has every word
		for(String key : tf.keySet()) {
			double tfVal = tf.getCount(key);
			double idfVal = idf.getCount(key);
			double tfidfVal = MathUtils.tdidf(tfVal, idfVal);
			tfidf.setCount(key, tfidfVal);
		}

		tfidf.keepTopNKeys(200);
		log.info("Tfidf keys " + tfidf + " got rid of " + (Math.abs(tfidf.size() - words.size())) + " words");

		for(String key : tfidf.getSortedKeys())
			vocab.add(key);


	}


	public void train() throws IOException {

		calcWordFrequencies();
		int size = vocab.size();
		//create a network that knows how to reconstruct the input documents
		if(classify) {
			cdbn = new TopicModelingCDBN.Builder()
			.numberOfInputs(size)
			.numberOfOutPuts(labels.size()).withActivation(new HardTanh())
			.hiddenLayerSizes(new int[]{size / 4,size / 8 ,numOuts})
			.build();
			discriminatory = new TopicModelingCDBN.Builder()
			.numberOfInputs(numOuts).withActivation(new HardTanh())
			.numberOfOutPuts(labels.size())
			.hiddenLayerSizes(new int[]{numOuts,size / 4,size / 4,size})
			.build();
		}

		//create a network that knows how to compress documents
		else
			cdbn = new TopicModelingCDBN.Builder()
		.numberOfInputs(vocab.size()).withActivation(new HardTanh())
		.numberOfOutPuts(labels.size())
		.hiddenLayerSizes( new int[]{size / 4,size / 8,numOuts})
		.build();


		List<DataSet> list = new ArrayList<DataSet>();

		for(File f : rootDir.listFiles())	 {

			for(File doc : f.listFiles()) {
				DoubleMatrix train = toWordCountVector(doc).transpose();

				DoubleMatrix outcome = MatrixUtil.toOutcomeVector(labels.indexOf(f.getName()), labels.size());
				list.add(new DataSet(train,outcome));

			}
		}

		DataSet data = DataSet.merge(list);
		Evaluation eval = new Evaluation();

		DoubleMatrix first = data.getFirst();
		//log.info("First " + first.toString("%.5f", "[","]", ",", "\n\n"));
		DoubleMatrix second = data.getSecond();
		cdbn.pretrain(first,1, 0.01, 1000);
		//cdbn.finetune(second, 0.1, 1000);
		trained = new DataSet(cdbn.reconstruct(first),second);


		if(classify) {
			DoubleMatrix reconstructed = cdbn.reconstruct(first);
			reconstructed = MatrixUtil.normalizeByRowSums(reconstructed);
			discriminatory.pretrain(reconstructed,1, 0.1, 1000);
			discriminatory.finetune(second, 0.01, 1000);
			eval.eval(second, discriminatory.predict(reconstructed));
			log.info("F - score so far " + eval.f1());
		}



		log.info("Final stats " + eval.stats());

		eval = new Evaluation();

		if(classify) {
			eval.eval(data.getSecond(), cdbn.predict(data.getFirst()));
			log.info("F - score for batch   after train is " + eval.f1());
		}




	}


	public void dump(String path) {
		File f = new File(path);
		try(BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(f,true))) {
			for(int i = 0; i < trained.numExamples(); i++) {
				int label = trained.get(i).outcome();
				DoubleMatrix input = trained.get(i).getFirst();
				StringBuffer line = new StringBuffer();
				for(int j = 0; j < input.length; j++) {
					line.append(input.get(j) + ",");

				}
				line.append(labels.get(label) + "\n");
				bos.write(line.toString().getBytes());
				bos.flush();
			}

			bos.close();
		}
		catch(Exception e) {
			throw new RuntimeException(e);
		}


	}

	/* Gather word count frequencies for a particular document */
	private DoubleMatrix toWordCountVector(File f) throws IOException {
		LineIterator iter = FileUtils.lineIterator(f);
		Counter<String> wordCounts = new Counter<String>();
		DoubleMatrix ret = new DoubleMatrix(vocab.size());
		while(iter.hasNext()) {
			String line = iter.nextLine();
			StringTokenizer tokenizer = new StringTokenizer(new InputHomogenization(line,true).transform());
			while(tokenizer.hasMoreTokens())  {
				String token = tokenizer.nextToken();
				if(!stopWords.contains(token)) {
					wordCounts.incrementCount(token,1.0);
				}

			}

		}

		for(int i = 0; i < vocab.size(); i++) {
			double count = wordCounts.getCount(vocab.get(i).toString());
			ret.put(i,count);
		}
		int nonStopWords = 0;
		for(int i = 0; i < ret.length; i++) {
			if(ret.get(i) > 0) {
				nonStopWords++;
			}
		}


		ret.divi(nonStopWords);

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
