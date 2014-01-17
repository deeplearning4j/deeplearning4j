package com.ccc.deeplearning.topicmodeling;

import java.io.File;
import java.io.IOException;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.StringTokenizer;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.LineIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.ccc.deeplearning.berkeley.Counter;
import com.ccc.deeplearning.berkeley.CounterMap;
import com.ccc.deeplearning.util.MathUtils;
import com.ccc.deeplearning.util.SetUtils;
import com.ccc.deeplearning.word2vec.ner.InputHomogenization;
import com.ccc.deeplearning.word2vec.viterbi.Index;

public class VocabCreator {

	private List<String> stopWords;
	public VocabCreator(List<String> stopWords, File rootDir) {
		super();
		this.stopWords = stopWords;
		this.rootDir = rootDir;
	}


	private File rootDir;
	private static Logger log = LoggerFactory.getLogger(VocabCreator.class);
	private CounterMap<String,String> words = new CounterMap<String,String>();
	private CounterMap<File,File> topics = new CounterMap<File,File>();
	private CounterMap<String,String> tf = new CounterMap<String,String>();
	private CounterMap<String,String> idf = new CounterMap<String,String>();
	private CounterMap<String,String> tfidf = new CounterMap<String,String>();



	public Index createVocab() throws IOException {
		Index vocab = new Index();

		//bootstrapping
		calcWordFrequencies();



		//term frequency has every word
		for(String topic : words.keySet()) {
			for(String word : words.getCounter(topic).keySet()) {
				double tfVal = tf.getCount(topic,word);
				double idfVal = idf.getCount(topic,word);
				double tfidfVal = MathUtils.tdidf(tfVal, idfVal);
				if(!stopWords.contains(word))
					tfidf.setCount(topic,word,tfidfVal);
			}


		}

		log.info("Tfidf keys " + tfidf + " got rid of " + (Math.abs(tfidf.size() - words.size())) + " words");
		/*
		 * Rank by tfidf of per topic.
		 * The topic/word pairs
		 * will make up the vocab
		 * .Rank by top n of each topic.
		 */
		Set<String> remove = new HashSet<String>();



		Counter<String> aggregate = new Counter<String>();

		for(String topic : tfidf.keySet()) {
			for(String word : tfidf.getCounter(topic).keySet()) {
				aggregate.setCount(word, tfidf.getCount(topic,word));
			}
		}


		for(String s : remove)
			aggregate.removeKey(s);




		aggregate.keepTopNKeys(500);

		log.info(tfidf.toString());

		for(String word : aggregate.keySet()) {
			if(vocab.indexOf(word) < 0)
				vocab.add(word);
		}


		return vocab;
	}


	/* initial statistics used for calculating word metadata such as word vectors to train on */
	private void calcWordFrequencies() throws IOException {
		for(File f : rootDir.listFiles())	 {
			for(File doc : f.listFiles()) {
				topics.incrementCount(f,doc,1.0);
				addForDocAndTopic(doc,f.getName());
			}
		} 
	}

	public double getCountForWord(String word) {
		double ret = 0.0;
		for(String topic : words.keySet()) {
			ret += words.getCount(topic, word);
		}

		return ret;
	}


	private void addForDocAndTopic(File doc,String topic) throws IOException {
		Set<String> encountered = new HashSet<String>();

		LineIterator iter = FileUtils.lineIterator(doc);
		while(iter.hasNext()) {
			String line = iter.nextLine();
			StringTokenizer tokenizer = new StringTokenizer(new InputHomogenization(line).transform());
			while(tokenizer.hasMoreTokens())  {
				String token = tokenizer.nextToken();
				if(!stopWords.contains(token)) {
					words.incrementCount(topic,token, 1.0);
					tf.incrementCount(topic,token, 1.0);
					if(!encountered.contains(token)) {
						idf.incrementCount(topic,token,1.0);
						encountered.add(token);
					}
				}

			}

		}	
	}


}
