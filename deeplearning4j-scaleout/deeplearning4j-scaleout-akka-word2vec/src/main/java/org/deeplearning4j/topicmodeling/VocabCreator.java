package org.deeplearning4j.topicmodeling;

import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.regex.Pattern;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.LineIterator;
import org.deeplearning4j.berkeley.Counter;
import org.deeplearning4j.berkeley.CounterMap;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.deeplearning4j.stopwords.StopWords;
import org.deeplearning4j.util.MathUtils;
import org.deeplearning4j.word2vec.inputsanitation.InputHomogenization;
import org.deeplearning4j.word2vec.sentenceiterator.LineSentenceIterator;
import org.deeplearning4j.word2vec.sentenceiterator.SentenceIterator;
import org.deeplearning4j.word2vec.tokenizer.DefaultTokenizerFactory;
import org.deeplearning4j.word2vec.tokenizer.Tokenizer;
import org.deeplearning4j.word2vec.tokenizer.TokenizerFactory;
import org.deeplearning4j.util.Index;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import akka.actor.ActorRef;
import akka.actor.ActorSystem;
import akka.actor.Props;
import akka.routing.RoundRobinPool;


/**
 * Sets up the word count frequencies
 * across a applyTransformToDestination of directories such that
 * the root directory has a directory for each topic
 * where the files underlying each directory
 * represent the topic.
 * @author Adam Gibson
 *
 */
public class VocabCreator implements Serializable {


	protected static final long serialVersionUID = -3006090530851522682L;
	//stop words loaded from class path
	protected List<String> stopWords;
	//cached vocab; usually used for serialization
	protected Index currVocab;
	//root directory of the topics
	protected File rootDir;
	protected static Logger log = LoggerFactory.getLogger(VocabCreator.class);
	//term frequency, how many times did the word occur overall?
	protected Counter<String> tf = new Counter<>();
	//inverse document frequency; number of documents a word occurs in within a dataset
	protected Counter<String> idf = new Counter<>();
	//ratio of number of times word occurs to the number of documents it occurs in
	protected Counter<String> tfidf = new Counter<>();
	protected Counter<String> wordScores;
	protected int numFiles;
	protected transient ActorSystem system;
	protected static Pattern punct = Pattern.compile("\\.?!:\\(\\);-',/\\[\\]~`@#$%^&*<>\"");
	//used for tracking metadata of document frequencies to words
	protected CounterMap<String,String> documentWordFrequencies = new CounterMap<String,String>();
	protected transient TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
	/**
	 * Creates a vocab based on tf-idf
	 * @param rootDir the root directory to iterate on
	 */
	public VocabCreator( File rootDir) {
		this(rootDir,new DefaultTokenizerFactory());
	}

	/**
	 * Creates a vocab based on tf-idf.
	 * This is a serialization constructor.
	 * Use at your own risk.
	 */
	public VocabCreator() {
	}


	/**
	 * Creates a vocab based on tf-idf
	 * @param rootDir the root directory to iterate on
	 * @param tokenizerFactory the tokenizer factory to use
	 */
	public VocabCreator(File rootDir,TokenizerFactory tokenizerFactory) {
		super();
		this.stopWords = StopWords.getStopWords();
		this.rootDir = rootDir;
		this.tokenizerFactory = tokenizerFactory;
		system = ActorSystem.create("WordFrequencySystem");
	}

	/* initial statistics used for calculating word metadata such as word vectors to iterate on */
	protected void calcWordFrequencies()  {
		numFiles = countFiles();
		CountDownLatch latch = new CountDownLatch(numFiles);
		ActorRef fileActor = system.actorOf(new RoundRobinPool(Runtime.getRuntime().availableProcessors()).props(Props.create(VocabCreatorActor.class,this,latch)));
		for(File f : rootDir.listFiles())	 {
			File[] subFiles = f.listFiles();
			if(f.isFile())
				fileActor.tell(f,fileActor);
			else if(subFiles != null)
				for(File doc : subFiles) {
					if(!doc.isFile())
						continue;

					fileActor.tell(doc,fileActor);
				}
		} 


		try {
			latch.await();
		} catch (InterruptedException e) {
			Thread.currentThread().interrupt();
		}

		log.info("Done calculating word frequencies");

	}

	protected int countFiles() {
		int ret = 0;
		for(File f : rootDir.listFiles())	 {
			File[] subFiles = f.listFiles();
			if(f.isFile())
				ret++;
			else if(subFiles != null)
				for(File doc : subFiles) {
					if(!doc.isFile())
						continue;

					ret++;
				}
		} 

		return ret;
	}

	/* should this word be included? */
	protected boolean validWord(String test) {
		return !stopWords.contains(test) && !punct.matcher(test).matches();
	}
	
	
	protected void addForDoc(File doc)  {
		Set<String> encountered = new HashSet<String>();
		SentenceIterator iter = new LineSentenceIterator(doc);
		while(iter.hasNext()) {
			String line = iter.nextSentence();
			if(line == null)
				continue;
			Tokenizer tokenizer = tokenizerFactory.create(new InputHomogenization(line).transform());
			while(tokenizer.hasMoreTokens())  {
				String token = tokenizer.nextToken();
				java.util.regex.Matcher m = punct.matcher(token);
				if(validWord(token)) {
					documentWordFrequencies.incrementCount(token,doc.getAbsolutePath(), 1.0);
					tf.incrementCount(token, 1.0);
					if(!encountered.contains(token)) {
						idf.incrementCount(token,1.0);
						encountered.add(token);
					}
				}

			}

			iter.finish();

		}

	}

	/**
	 * Creates an index of the top size words
	 * based on tf-idf metrics
	 * @param size the number of words in the vocab
	 * @return the index of the words
	 * @throws IOException
	 */
	public Index createVocab(int size)  {
		Index vocab = new Index();

		//bootstrapping
		calcWordFrequencies();



		//term frequency has every word
		for(String word : tf.keySet()) {
			double tfVal = MathUtils.tf((int) documentWordFrequencies.getCount(word));
			double idfVal = MathUtils.idf(numFiles, idf.getCount(word));
			double tfidfVal = MathUtils.tfidf(tfVal, idfVal);
			java.util.regex.Matcher m = punct.matcher(word);

			if(!stopWords.contains(word) && !m.matches())
				tfidf.setCount(word,tfidfVal);
		}







		Counter<String> aggregate = tfidf;


		//keep top size keys via tfidf rankings
		aggregate.keepTopNKeys(size - 1);

		log.info("Created vocab of size " + aggregate.size());

		wordScores = aggregate;

		//add words that made it via rankings
		for(String word : aggregate.keySet()) {
			if(vocab.indexOf(word) < 0)
				vocab.add(word);
		}

		//cache the vocab
		currVocab = vocab;
		return vocab;
	}


	public INDArray getScoreMatrix(File file) {
		Counter<String> docWords = new Counter<String>();
		try {
			LineIterator iter = FileUtils.lineIterator(file);
			while(iter.hasNext()) {
				Tokenizer t = tokenizerFactory.create((new InputHomogenization(iter.nextLine()).transform()));
				while(t.hasMoreTokens()) {
					docWords.incrementCount(t.nextToken(), 1.0);
				}
			}

			iter.close();
		} catch (IOException e) {
			throw new IllegalStateException("Unable to read file",e);
		}
		INDArray ret = Nd4j.create(1, currVocab.size());


		for(int i = 0; i < currVocab.size(); i++) {
			if(docWords.getCount(currVocab.get(i).toString()) > 0) {
				ret.putScalar(i,wordScores.getCount(currVocab.get(i).toString()));
			}
		}

		return ret;
	}



	public synchronized List<String> getStopWords() {
		return stopWords;
	}



	public synchronized void setStopWords(List<String> stopWords) {
		this.stopWords = stopWords;
	}



	public synchronized File getRootDir() {
		return rootDir;
	}



	public synchronized void setRootDir(File rootDir) {
		this.rootDir = rootDir;
	}


	public synchronized Counter<String> getTf() {
		return tf;
	}



	public synchronized void setTf(Counter<String> tf) {
		this.tf = tf;
	}



	public synchronized Counter<String> getIdf() {
		return idf;
	}



	public synchronized void setIdf(Counter<String> idf) {
		this.idf = idf;
	}



	public synchronized Counter<String> getTfidf() {
		return tfidf;
	}



	public synchronized void setTfidf(Counter<String> tfidf) {
		this.tfidf = tfidf;
	}

	public synchronized Counter<String> getWordScores() {
		return wordScores;
	}



	public synchronized void setWordScores(Counter<String> wordScores) {
		this.wordScores = wordScores;
	}





	public double getCountForWord(String word) {
		return wordScores.getCount(word);
	}

	public Index getCurrVocab() {
		return currVocab;
	}




}
