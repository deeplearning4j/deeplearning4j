package org.deeplearning4j.word2vec;

import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Set;
import java.util.Stack;
import java.util.TreeSet;
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

import org.apache.commons.io.IOUtils;
import org.deeplearning4j.berkeley.Counter;
import org.deeplearning4j.berkeley.Triple;
import org.deeplearning4j.nn.Persistable;
import org.deeplearning4j.util.MatrixUtil;
import org.deeplearning4j.word2vec.actor.SentenceActor;
import org.deeplearning4j.word2vec.actor.VocabActor;
import org.deeplearning4j.word2vec.sentenceiterator.CollectionSentenceIterator;
import org.deeplearning4j.word2vec.sentenceiterator.SentenceIterator;
import org.deeplearning4j.word2vec.tokenizer.DefaultTokenizerFactory;
import org.deeplearning4j.word2vec.tokenizer.Tokenizer;
import org.deeplearning4j.word2vec.tokenizer.TokenizerFactory;
import org.deeplearning4j.word2vec.util.Util;
import org.deeplearning4j.word2vec.viterbi.Index;
import org.jblas.DoubleMatrix;
import org.jblas.SimpleBlas;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import scala.concurrent.Future;
import akka.actor.ActorRef;
import akka.actor.ActorSystem;
import akka.actor.Props;
import akka.dispatch.Futures;
import akka.dispatch.OnComplete;
import akka.routing.RoundRobinPool;


/**
 * Leveraging a 3 layer neural net with a softmax approach as output,
 * converts a word based on its context and the training examples in to a
 * numeric vector
 * @author Adam Gibson
 *
 */
public class Word2Vec implements Persistable {


	private static final long serialVersionUID = -2367495638286018038L;
	private Map<String,VocabWord> vocab = new ConcurrentHashMap<String,VocabWord>();

	private transient TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
	private transient SentenceIterator sentenceIter;
	private int topNSize = 40;
	//matrix row of a given word
	private Index wordIndex = new Index();
	private int sample = 1;
	//learning rate
	private Double alpha = 0.025;
	private int wordCount  = 0;
	public final static double MIN_ALPHA =  0.001;
	//number of times the word must occur in the vocab to appear in the calculations, otherwise treat as unknown
	private int minWordFrequency = 5;
	//context to use for gathering word frequencies
	private int window = 5;
	private int trainWordsCount = 0;
	//number of neurons per layer
	private int layerSize = 50;
	private static Logger log = LoggerFactory.getLogger(Word2Vec.class);
	private int size = 0;
	private int words = 0;
	//input layer
	private DoubleMatrix syn0,syn0Norm;
	//hidden layer
	private DoubleMatrix syn1;
	private int allWordsCount = 0;
	private int numSentencesProcessed = 0;
	private static ActorSystem trainingSystem;
	private List<String> stopWords;
	/* out of vocab */
	private double[] oob;
	private boolean shouldReset = true;

	public Word2Vec() {}

	/**
	 * Specify a sentence iterator
	 * 
	 * 
	 * 
	 * 
	 *
	 */
	public Word2Vec(SentenceIterator sentenceIter) {
		oob = new double[layerSize];
		Arrays.fill(oob,0.0);
		readStopWords();
		this.sentenceIter = sentenceIter;
		buildVocab();
	}



	public Word2Vec(SentenceIterator sentenceIter,int minWordFrequency) {
		oob = new double[layerSize];
		Arrays.fill(oob,0.0);
		readStopWords();
		this.sentenceIter = sentenceIter;
		this.minWordFrequency = minWordFrequency;
	}


	public Word2Vec(TokenizerFactory factory,SentenceIterator sentenceIter) {
		this(sentenceIter);
		this.tokenizerFactory = factory;
	}

	/**
	 * Specify a custom tokenizer, sentence iterator
	 * and minimum word frequency
	 * @param factory
	 * @param sentenceIter
	 * @param minWordFrequency
	 */
	public Word2Vec(TokenizerFactory factory,SentenceIterator sentenceIter,int minWordFrequency) {
		this(factory,sentenceIter);
		this.minWordFrequency = minWordFrequency;
	}



	/**
	 * Assumes whole dataset is passed in. 
	 * This is purely meant for batch methods.
	 * Same as calling {@link #Word2Vec(Collection, int)}
	 * with the second argument being 5
	 * @param sentences the sentences to use
	 * to train on
	 */
	public Word2Vec(Collection<String> sentences) {
		this(sentences,5);
		readStopWords();

	}

	public Word2Vec(Collection<String> sentences,TokenizerFactory factory) {
		this(sentences);
		this.tokenizerFactory = factory;
	}

	/**
	 * Initializes based on assumption of whole data set being passed in.
	 * @param sentences the sentences to be used for training
	 * @param minWordFrequency the minimum word frequency
	 * to be counted in the vocab
	 */
	public Word2Vec(Collection<String> sentences,int minWordFrequency) {
		this.minWordFrequency = minWordFrequency;
		this.sentenceIter = new CollectionSentenceIterator(sentences);

		this.buildVocab();

		oob = new double[layerSize];
		Arrays.fill(oob,0.0);
		readStopWords();

	}


	public Word2Vec(Collection<String> sentences,int minWordFrequency,TokenizerFactory factory) {
		this(sentences,minWordFrequency);
		this.tokenizerFactory = factory;
	}


	public double[] getWordVectorNormalized(String word) {
		int i = this.wordIndex.indexOf(word);
		if(i < 0) {
			i = wordIndex.indexOf("STOP");
			if(i < 0)
				return oob;
		}

		return syn0Norm.getRow(i).toArray();
	}

	public double[] getWordVector(String word) {
		int i = this.wordIndex.indexOf(word);
		if(i < 0) {
			i = wordIndex.indexOf("STOP");
			if(i < 0)
				return oob;
		}

		return syn0.getRow(i).toArray();
	}

	public int indexOf(String word) {
		return wordIndex.indexOf(word);
	}

	public DoubleMatrix getWordVectorMatrix(String word) {
		int i = this.wordIndex.indexOf(word);
		if(i < 0)
			return new DoubleMatrix(oob);
		return syn0.getRow(i);
	}

	public DoubleMatrix getWordVectorMatrixNormalized(String word) {
		int i = this.wordIndex.indexOf(word);
		if(i < 0)
			return new DoubleMatrix(oob);
		return syn0.getRow(i);
	}



	public VocabWord getWord(String key) {
		return vocab.get(key);
	}





	public Collection<String> wordsNearest(String word,int n) {
		DoubleMatrix vec = this.getWordVectorMatrix(word);
		if(vec == null)
			return new ArrayList<String>();
		Counter<String> distances = new Counter<String>();
		for(int i = 0; i < syn0.rows; i++) {
			double sim = similarity(word,wordIndex.get(i).toString());
			distances.incrementCount(wordIndex.get(i).toString(), sim);
		}


		distances.keepTopNKeys(n);
		return distances.keySet();

	}


	public List<String> analogyWords(String w1,String w2,String w3) {
		TreeSet<VocabWord> analogies = this.analogy(w1, w2, w3);
		List<String> ret = new ArrayList<String>();
		for(VocabWord w : analogies)
			ret.add(wordIndex.get(w.getIndex()).toString());
		return ret;
	}




	private void insertTopN(String name, double score, List<VocabWord> wordsEntrys) {
		if (wordsEntrys.size() < topNSize) {
			VocabWord v = new VocabWord(score,layerSize);
			v.setIndex(wordIndex.indexOf(name));
			wordsEntrys.add(v);
			return;
		}
		double min = Float.MAX_VALUE;
		int minOffe = 0;
		int minIndex = -1;
		for (int i = 0; i < topNSize; i++) {
			VocabWord wordEntry = wordsEntrys.get(i);
			if (min > wordEntry.getWordFrequency()) {
				min = (double) wordEntry.getWordFrequency();
				minOffe = i;
				minIndex = wordEntry.getIndex();
			}
		}

		if (score > min) {
			VocabWord w = new VocabWord(score,layerSize);
			w.setIndex(minIndex);
			wordsEntrys.set(minOffe,w);
		}

	}

	public boolean hasWord(String word) {
		return wordIndex.indexOf(word) >= 0;
	}

	public void train() {
		if(trainingSystem == null)
			trainingSystem = ActorSystem.create();

		if(stopWords == null)
			readStopWords();
		log.info("Training word2vec multithreaded");

		

		final Counter<String> totalWords = Util.parallelCounter();

		getSentenceIter().reset();

		final AtomicLong changed = new AtomicLong(System.currentTimeMillis());


		ActorRef sentenceActor  = trainingSystem.actorOf(new RoundRobinPool(Runtime.getRuntime().availableProcessors() *3 ).props(Props.create(new SentenceActor.SentenceActorCreator(this)).withDispatcher("akka.actor.worker-dispatcher")));


		if(syn0.rows != this.vocab.size())
			throw new IllegalStateException("We appear to be missing vectors here. Unable to train. Please ensure vectors were loaded properly.");

		int numSentences = 0;

		while(getSentenceIter().hasNext()) {
			final String sentence = sentenceIter.nextSentence();
			if(sentence != null) {
				Future<Void> f = Futures.future(new Callable<Void>() {

					@Override
					public Void call() throws Exception {
						processSentence(sentence, totalWords);
						return null;
					}

				},trainingSystem.dispatcher());
				f.onComplete(new OnComplete<Void>() {

					@Override
					public void onComplete(Throwable arg0, Void arg1)
							throws Throwable {
						if(arg0 != null)
							throw arg0;
						numSentencesProcessed++;
						changed.set(System.currentTimeMillis());

					}

				}, trainingSystem.dispatcher());

			}

			/*
			sentenceActor.tell(new SentenceMessage(totalWords, sentence, changed),sentenceActor);
			numSentences++;
			if(numSentences % 10000 == 0) {
				log.info("Sent " + numSentences + " for training");
			}*/
		}


		boolean done = false;
		long fiveMinutes = TimeUnit.MINUTES.toMillis(1);
		while(!done) {
			long curr = System.currentTimeMillis();
			long lastChanged = changed.get();
			long diff = Math.abs(curr - lastChanged);
			//hasn't changed for 5 minutes
			if(diff >= fiveMinutes) {
				done = true;	
			} 

			else
				try {
					Thread.sleep(15000);
				} catch (InterruptedException e) {
					Thread.currentThread().interrupt();
				}
		}


		log.info("Shutting down system; done training");

		if(trainingSystem != null)
			trainingSystem.shutdown();

	}

	public void processSentence(final String sentence,final Counter<String> totalWords) {
		trainSentence(sentence, totalWords);
		if(numSentencesProcessed % 10000 == 0) {
			alpha = new Double(Math.max(MIN_ALPHA, alpha * (1 - 1.0 * totalWords.totalCount() / allWordsCount)));
			log.info("Alpha updated " + alpha + " progress " + numSentencesProcessed);
		}
	}




	public List<VocabWord> trainSentence(String sentence,Counter<String> totalWords) {
		Tokenizer tokenizer = tokenizerFactory.create(sentence);
		List<VocabWord> sentence2 = new ArrayList<VocabWord>();
		while(tokenizer.hasMoreTokens()) {
			String next = tokenizer.nextToken();
			if(stopWords.contains(next))
				next = "STOP";
			VocabWord word = vocab.get(next);
			if(word == null) 
				continue;

			sentence2.add(word);
			totalWords.incrementCount(next, 1.0);

		}

		trainSentence(sentence2);
		return sentence2;
	}


	/**
	 *
	 * 
	 * @param word
	 * @return
	 */
	public Set<VocabWord> distance(String word) {
		DoubleMatrix wordVector = getWordVectorMatrix(word);
		if (wordVector == null) {
			return null;
		}
		DoubleMatrix tempVector = null;
		List<VocabWord> wordEntrys = new ArrayList<VocabWord>(topNSize);
		String name = null;
		for (int i = 0; i < syn0.rows; i++) {
			name = wordIndex.get(i).toString();
			if (name.equals(word)) {
				continue;
			}
			double dist = 0;
			tempVector = syn0.getRow(i);
			dist = wordVector.dot(tempVector);
			insertTopN(name, dist, wordEntrys);
		}
		return new TreeSet<VocabWord>(wordEntrys);
	}

	/**
	 *
	 * @return 
	 */
	public TreeSet<VocabWord> analogy(String word0, String word1, String word2) {
		DoubleMatrix wv0 = getWordVectorMatrix(word0);
		DoubleMatrix wv1 = getWordVectorMatrix(word1);
		DoubleMatrix wv2 = getWordVectorMatrix(word2);


		DoubleMatrix wordVector = wv1.sub(wv0).add(wv2);

		if (wv1 == null || wv2 == null || wv0 == null) 
			return null;

		DoubleMatrix tempVector;
		String name;
		List<VocabWord> wordEntrys = new ArrayList<VocabWord>(topNSize);
		for (int i = 0; i < syn0.rows; i++) {
			name = wordIndex.get(i).toString();

			if (name.equals(word0) || name.equals(word1) || name.equals(word2)) {
				continue;
			}
			tempVector = syn0.getRow(i);
			double dist = wordVector.dot(tempVector);
			insertTopN(name, dist, wordEntrys);
		}
		return new TreeSet<VocabWord>(wordEntrys);
	}


	public void setup() {

		log.info("Building binary tree");
		buildBinaryTree();
		log.info("Resetting weights");
		if(shouldReset)
			resetWeights();
	}


	public void buildVocab() {
		readStopWords();

		if(trainingSystem == null)
			trainingSystem = ActorSystem.create();



		final Counter<String> rawVocab = Util.parallelCounter();
		final AtomicLong semaphore = new AtomicLong(System.currentTimeMillis());
		final AtomicInteger numSentences = new AtomicInteger(0);
		int queued = 0;

		final ActorRef vocabActor = trainingSystem.actorOf(new RoundRobinPool(Runtime.getRuntime().availableProcessors()).props(Props.create(VocabActor.class,tokenizerFactory,wordIndex,minWordFrequency,vocab,layerSize,stopWords,rawVocab,semaphore)));

		/* all words; including those not in the actual ending index */
		while(getSentenceIter().hasNext()) {
			String sentence = getSentenceIter().nextSentence();
			if(sentence == null)
				continue;

			vocabActor.tell(sentence, vocabActor);
			log.info("Sent " + queued);
			queued++;



		}

		boolean done = false;
		long fiveMinutes = TimeUnit.MINUTES.toMillis(1);
		while(!done) {
			long curr = System.currentTimeMillis();
			long lastChanged = semaphore.get();
			long diff = Math.abs(curr - lastChanged);
			log.info("Waiting on setup...");
			//hasn't changed for 5 minutes
			if(diff >= fiveMinutes) {
				done = true;	
			} 

			else
				try {
					Thread.sleep(15000);
				} catch (InterruptedException e) {
					Thread.currentThread().interrupt();
				}
		}


		setup();

	}

	public void trainSentence(List<VocabWord> sentence) {
		long nextRandom = 5;
		for(int i = 0; i < sentence.size(); i++) {
			VocabWord entry = sentence.get(i);
			// The subsampling randomly discards frequent words while keeping the ranking same
			if (sample > 0) {
				double ran = (Math.sqrt(entry.getWordFrequency() / (sample * trainWordsCount)) + 1)
						* (sample * trainWordsCount) / entry.getWordFrequency();
				nextRandom = nextRandom * 25214903917L + 11;
				if (ran < (nextRandom & 0xFFFF) / (double) 65536) {
					continue;
				}
			}
			nextRandom = nextRandom * 25214903917L + 11;
			int b = (int) nextRandom % window;
			skipGram(i,sentence,b);
		}
	}



	public void skipGram(int i,List<VocabWord> sentence,int b) {
		VocabWord word = sentence.get(i);
		if(word == null)
			return;


		//subsampling
		for(int j = b; j < window * 2 + 1 - b; j++) {
			if(j == window)
				continue;
			int c1 = i - window + j;

			if (c1 < 0 || c1 >= sentence.size()) 
				continue;

			VocabWord word2 = sentence.get(c1);
			iterate(word,word2);
		}
	}

	public void  iterate(VocabWord w1,VocabWord w2) {
		DoubleMatrix l1 = syn0.getRow(w2.getIndex());
		DoubleMatrix l2a = syn1.getRows(w1.getCodes());
		DoubleMatrix fa = MatrixUtil.sigmoid(MatrixUtil.dot(l1, l2a.transpose()));
		// ga = (1 - word.code - fa) * alpha  # vector of error gradients multiplied by the learning rate
		DoubleMatrix ga = DoubleMatrix.ones(fa.length).sub(MatrixUtil.toMatrix(w1.getCodes())).sub(fa).mul(alpha);
		DoubleMatrix outer = ga.mmul(l1);
		for(int i = 0; i < w1.getPoints().length; i++) {
			DoubleMatrix toAdd = l2a.getRow(i).add(outer.getRow(i));
			syn1.putRow(w1.getPoints()[i],toAdd);
		}

		DoubleMatrix updatedInput = l1.add(MatrixUtil.dot(ga, l2a));
		syn0.putRow(w2.getIndex(),updatedInput);
	}




	/* Builds the binary tree for the word relationships */
	private void buildBinaryTree() {
		PriorityQueue<VocabWord> heap = new PriorityQueue<VocabWord>(vocab.values());
		int i = 0;
		while(heap.size() > 1) {
			VocabWord min1 = heap.poll();
			VocabWord min2 = heap.poll();


			VocabWord add = new VocabWord(min1.getWordFrequency() + min2.getWordFrequency(),layerSize);
			int index = (vocab.size() + i);

			add.setIndex(index); 
			add.setLeft(min1);
			add.setRight(min2);
			min1.setCode(0);
			min2.setCode(1);
			min1.setParent(add);
			min2.setParent(add);
			heap.add(add);
			i++;
		}

		Triple<VocabWord,int[],int[]> triple = new Triple<VocabWord,int[],int[]>(heap.poll(),new int[]{},new int[]{});
		Stack<Triple<VocabWord,int[],int[]>> stack = new Stack<>();
		stack.add(triple);
		while(!stack.isEmpty())  {
			triple = stack.pop();
			int[] codes = triple.getSecond();
			int[] points = triple.getThird();
			VocabWord node = triple.getFirst();
			if(node == null) {
				log.info("Node was null");
				continue;
			}
			if(node.getIndex() < vocab.size()) {
				node.setCodes(codes);
				node.setPoints(points);
			}
			else {
				int[] copy = plus(points,node.getIndex() - vocab.size());
				points = copy;
				triple.setThird(points);
				stack.add(new Triple<VocabWord,int[],int[]>(node.getLeft(),plus(codes,0),points));
				stack.add(new Triple<VocabWord,int[],int[]>(node.getRight(),plus(codes,1),points));

			}
		}



		log.info("Built tree");
	}

	private int[] plus (int[] addTo,int add) {
		int[] copy = new int[addTo.length + 1];
		for(int c = 0; c < addTo.length; c++)
			copy[c] = addTo[c];
		copy[addTo.length] = add;
		return copy;
	}


	/* reinit weights */
	private void resetWeights() {
		syn1 = DoubleMatrix.zeros(vocab.size(), layerSize);
		syn0 = DoubleMatrix.zeros(vocab.size(),layerSize);
		org.jblas.util.Random.seed(1);
		for(int i = 0; i < syn0.rows; i++)
			for(int j = 0; j < syn0.columns; j++) {
				syn0.put(i,j,(org.jblas.util.Random.nextDouble() - 0.5) / layerSize);
			}
	}


	/**
	 * Returns the similarity of 2 words
	 * @param word the first word
	 * @param word2 the second word
	 * @return a normalized similarity (cosine similarity)
	 */
	public double similarity(String word,String word2) {
		if(word.equals(word2))
			return 1.0;
		if(syn0Norm == null)
			this.syn0Norm = syn0.div(SimpleBlas.nrm2(syn0));
		DoubleMatrix vector = getWordVectorMatrixNormalized(word);
		DoubleMatrix vector2 = getWordVectorMatrixNormalized(word2);
		if(vector == null || vector2 == null)
			return -1;
		DoubleMatrix d1 = MatrixUtil.unitVec(vector);
		DoubleMatrix d2 = MatrixUtil.unitVec(vector2);
		double ret = d1.dot(d2);
		if(ret <  0)
			return 0;
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


	public void setMinWordFrequency(int minWordFrequency) {
		this.minWordFrequency = minWordFrequency;
	}


	public int getLayerSize() {
		return layerSize;
	}
	public void setLayerSize(int layerSize) {
		this.layerSize = layerSize;
	}


	public int getTrainWordsCount() {
		return trainWordsCount;
	}


	public Index getWordIndex() {
		return wordIndex;
	}
	public void setWordIndex(Index wordIndex) {
		this.wordIndex = wordIndex;
	}


	public DoubleMatrix getSyn0() {
		return syn0;
	}

	public DoubleMatrix getSyn1() {
		return syn1;
	}


	public Map<String, VocabWord> getVocab() {
		return vocab;
	}

	public double getAlpha() {
		return alpha;
	}




	public int getWordCount() {
		return wordCount;
	}




	public int getMinWordFrequency() {
		return minWordFrequency;
	}




	public int getWindow() {
		return window;
	}


	public int getTopNSize() {
		return topNSize;
	}

	public int getSample() {
		return sample;
	}



	public int getSize() {
		return size;
	}

	public double[]	 getOob() {
		return oob;
	}

	public int getWords() {
		return words;
	}



	public int getAllWordsCount() {
		return allWordsCount;
	}

	public static ActorSystem getTrainingSystem() {
		return trainingSystem;
	}


	public void setSyn0(DoubleMatrix syn0) {
		this.syn0 = syn0;
	}

	public void setSyn1(DoubleMatrix syn1) {
		this.syn1 = syn1;
	}

	public void setWindow(int window) {
		this.window = window;
	}




	public List<String> getStopWords() {
		return stopWords;
	}

	public synchronized SentenceIterator getSentenceIter() {
		return sentenceIter;
	}



	public synchronized TokenizerFactory getTokenizerFactory() {
		return tokenizerFactory;
	}

	/**
	 * Note that calling a setter on this
	 * means assumes that this is a training continuation
	 * and therefore weights should not be reset.
	 * @param sentenceIter
	 */
	public void setSentenceIter(SentenceIterator sentenceIter) {
		this.sentenceIter = sentenceIter;
		this.shouldReset = false;
	}

	@Override
	public void write(OutputStream os) {
		try {
			ObjectOutputStream dos = new ObjectOutputStream(os);

			dos.writeObject(this);


		} catch (IOException e) {
			throw new RuntimeException(e);
		}

	}

	@Override
	public void load(InputStream is) {
		try {
			ObjectInputStream ois = new ObjectInputStream(is);
			Word2Vec vec = (Word2Vec) ois.readObject();
			this.allWordsCount = vec.allWordsCount;
			this.alpha = vec.alpha;
			this.minWordFrequency = vec.minWordFrequency;
			this.numSentencesProcessed = vec.numSentencesProcessed;
			this.oob = vec.oob;
			this.sample = vec.sample;
			this.size = vec.size;
			this.wordIndex = vec.wordIndex;
			this.stopWords = vec.stopWords;
			this.syn0 = vec.syn0;
			this.syn1 = vec.syn1;
			this.topNSize = vec.topNSize;
			this.trainWordsCount = vec.trainWordsCount;
			this.window = vec.window;

		}catch(Exception e) {
			throw new RuntimeException(e);
		}



	}


}
