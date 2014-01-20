package com.ccc.deeplearning.topicmodeling;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.apache.commons.io.LineIterator;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import com.ccc.deeplearning.autoencoder.DeepAutoEncoder;
import com.ccc.deeplearning.berkeley.Counter;
import com.ccc.deeplearning.datasets.DataSet;
import com.ccc.deeplearning.dbn.CDBN;
import com.ccc.deeplearning.dbn.DBN;
import com.ccc.deeplearning.eval.Evaluation;
import com.ccc.deeplearning.nn.BaseMultiLayerNetwork;
import com.ccc.deeplearning.nn.activation.HardTanh;
import com.ccc.deeplearning.nn.activation.Tanh;
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
	private TopicModelingCDBN cdbn;
	private DeepAutoEncoder autoEncoder;
	private Index vocab = new Index();
	private List<String> stopWords;
	private File rootDir;
	private int numOuts;
	private DataSet trained;
	private VocabCreator vocabCreator;

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
	public TopicModeling(File rootDir,int numOuts) {
		this.numOuts = numOuts;
		this.rootDir = rootDir;
		File[] subs = rootDir.listFiles();

		labels = new ArrayList<String>(subs.length);
		for(File f : subs) {
			labels.add(f.getName());
		}
		readStopWords();

	}






	public void train() throws IOException {
		this.vocabCreator = new VocabCreator(stopWords,rootDir);
		vocab = vocabCreator.createVocab();

		int size = vocab.size();
		//create a network that knows how to reconstruct the input documents

		cdbn = (TopicModelingCDBN) new TopicModelingCDBN.Builder()
		.numberOfInputs(size)
		.numberOfOutPuts(labels.size())
		.hiddenLayerSizes( new int[]{size, size, size})
		.build();


		List<DataSet> list = new ArrayList<DataSet>();
		File[] dirs =  rootDir.listFiles();

		for(File f : dirs)	 {

			for(File doc : f.listFiles()) {
				DoubleMatrix train = toWordCountVector(doc).transpose();

				DoubleMatrix outcome = MatrixUtil.toOutcomeVector(labels.indexOf(f.getName()), labels.size());
				//DoubleMatrix outcome = DoubleMatrix.scalar(labels.indexOf(f.getName()));
				for(int i = 0; i < 100;i++)
					list.add(new DataSet(train,outcome));



			}
		}



		DataSet data = DataSet.merge(list);
		Evaluation eval = new Evaluation();

		DoubleMatrix first = data.getFirst();
		//log.info("First " + first.toString("%.5f", "[","]", ",", "\n\n"));
		DoubleMatrix second = data.getSecond();
		/*	autoEncoder = new DeepAutoEncoder(cdbn,new Object[]{1,0.1,1000});
		autoEncoder.train(first, second, 0.1);

		log.info("Predicting " + first + "\n" + " with labels " + second);
		 */
		//cdbn.pretrain(first,0.01, 0.5, 1000);
		cdbn.pretrain(first, 1,0.1,1000);
		cdbn.finetune(second, 0.1, 1000);
		DoubleMatrix predicted = cdbn.predict(first);
		//cdbn.finetune(second, 0.1, 1000);
		DoubleMatrix outcomes2 = MatrixUtil.outcomes(predicted);
		//log.info("Predicted was " + outcomes2);
		//log.info("Actual was " + second);

		eval.eval(second,predicted);
		log.info("Final stats for first " + eval.stats());
		/*

		DoubleMatrix searched = autoEncoder.encode(first);
		Evaluation eval2 = new Evaluation();
		 */


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

	/*	for(int i = 0; i < ret.length; i++) {
			if(ret.get(i) > 0)
				ret.put(i,1);
		}*/


		ret.divi(nonStopWords);
		double sum = ret.sum();
		return ret;
	}


	/* Compression/reconstruction */
	public DoubleMatrix reconstruct(String document,int layer) {
		return cdbn.reconstruct(toDocumentMatrix(document),layer);
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
					ret.put(idx, vocabCreator.getCountForWord(token));
				}
			}

		}
		//normalize the word counts by the number of words
		ret.divi(ret.length);

		/*for(int i = 0; i < ret.length; i++) {
			if(ret.get(i) > 0)
				ret.put(i,1);
		}*/


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
