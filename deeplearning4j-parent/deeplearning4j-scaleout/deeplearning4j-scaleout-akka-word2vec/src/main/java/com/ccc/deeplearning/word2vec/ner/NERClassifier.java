package com.ccc.deeplearning.word2vec.ner;


import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;

import org.apache.commons.math3.random.MersenneTwister;
import org.jblas.DoubleMatrix;
import org.jblas.SimpleBlas;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.ccc.deeplearning.berkeley.CounterMap;
import com.ccc.deeplearning.datasets.DataSet;
import com.ccc.deeplearning.dbn.CDBN;
import com.ccc.deeplearning.nn.BaseMultiLayerNetwork;
import com.ccc.deeplearning.word2vec.Word2Vec;
import com.ccc.deeplearning.word2vec.nn.multilayer.Word2VecMultiLayerNetwork;
import com.ccc.deeplearning.word2vec.util.Window;
import com.ccc.deeplearning.word2vec.util.WindowConverter;
import com.ccc.deeplearning.word2vec.util.Windows;
import com.ccc.deeplearning.word2vec.viterbi.CounterUtil;
import com.ccc.deeplearning.word2vec.viterbi.Datum;
import com.ccc.deeplearning.word2vec.viterbi.Index;
import com.ccc.deeplearning.word2vec.viterbi.Viterbi;
import com.ccc.deeplearning.word2vec.viterbi.ViterbiUtil;

/**
 * Trains a named entity recognition classifier
 * based on word2vec. Training examples are read
 * by files and labelStrings are in plain text files
 * with the form: <LABEL> positive example </LABEL>
 * @author Adam Gibson
 *
 */
public class NERClassifier implements Serializable {

	private static final long serialVersionUID = -7010410698000526588L;
	private Word2Vec vec;
	//training vectors
	private List<double[]> trainingExamples = new ArrayList<double[]>();
	//windows for training examples
	private List<Window> exampleString	= new ArrayList<Window>();
	//vectorized outcome (1,0), (0,1)
	private List<double[]> trainingOutput = new ArrayList<double[]>();
	private static Logger log = LoggerFactory.getLogger(NERClassifier.class);
	private Word2VecMultiLayerNetwork prop;
	//string labelStrings for each training example
	private List<String> stringOutcomes = new ArrayList<String>();
	//available labelStrings
	private List<String> labels;
	private Viterbi viterbi;
	private Object[] deepLearningParams;
	//depending if one is specified should train or not
	private boolean trainWord2Vec;
	private boolean trainDeepNet;
	
	
	public NERClassifier(Word2VecMultiLayerNetwork network,Word2Vec vec,Viterbi viterbi,List<String> labels) {
		this.prop = network;
		this.vec = vec;
		this.viterbi = viterbi;
		this.labels = labels;
	}
	
	/**
	 * Creates a classifier
	 * This will also train a word2vec model based on the 
	 * training examples
	 * @param labelStrings the possible outcomes
	 */
	public NERClassifier(List<String> labels) {
		this(labels,null);

	}
	/**
	 * Creates a classifier
	 * This will also train a word2vec model based on the 
	 * training examples
	 * @param labelStrings the possible outcomes
	 */
	public NERClassifier(List<String> labels,Object[] deepLearningParams) {
		this.vec = new Word2Vec();
		this.labels = labels;
		if(deepLearningParams != null)
			this.deepLearningParams = deepLearningParams;
		else {
			this.deepLearningParams = new Object[]{1,0.1,50};
		}		
		trainWord2Vec = true;
		trainDeepNet = true;
		//already counted; therefore redundant
		if(this.labels.contains("NONE")) {
			this.labels.remove("NONE");
			log.warn("No need to include NONE as a label, taken care of internally; removing");
		}

	}
	/**
	 * Creates a classifier with the specified word2vec as training examples
	 * @param labelStrings
	 * @param vec
	 */
	public NERClassifier(List<String> labels,Word2Vec vec,Object[] deepLearningParams) {
		this.vec = vec;
		this.labels = labels;
		trainWord2Vec = false;
		trainDeepNet = true;
		if(deepLearningParams != null)
			this.deepLearningParams = deepLearningParams;
		else {
			this.deepLearningParams = new Object[]{1,0.1,50};
		}
		//already counted; therefore redundant
		if(this.labels.contains("NONE")) {
			this.labels.remove("NONE");
			log.warn("No need to include NONE as a label, taken care of internally; removing");
		}

	}


	/* CREATE SOMETHING TO MAP INPUT TRAINING DATA WITH A CONTEXT MAPPING TO INPUT DATA */
	public double[][] getInputs() {
		double[][] input = new double[trainingExamples.size()][trainingExamples.get(0).length];
		for(int i = 0; i < trainingExamples.size(); i++) 
			for(int j = 0; j < trainingExamples.get(i).length; j++)
				input[i][j] = trainingExamples.get(i)[j];


		return input;
	}




	/* CREATE SOMETHING TO MAP LABELED TRAINING OUTPUTS TO DATA */
	public double[][] getOutputs() {
		double[][] ret = new double[trainingOutput.size()][labels.size() + 1];

		for (int i = 0; i < trainingOutput.size(); i++) {
			ret[i] = trainingOutput.get(i);
		} 

		return ret;
	}

	public void addExample(String training) {
		if(training == null || training.isEmpty())
			return;
		List<String> words = new ArrayList<String>();
		StringTokenizer tokenizer = new StringTokenizer(new InputHomogenization(training).transform());
		while(tokenizer.hasMoreTokens()) {
			String token = tokenizer.nextToken();
			//training label
			if(token.charAt(0) == '<' && token.endsWith(">"))
				token = token.toUpperCase();
			words.add(token);
		}
		String currLabel = "NONE";

		List<Window> windows = Windows.windows(words,vec.getWindow());

		for(Window window : windows) {
			if(window.isBeginLabel())
				currLabel = window.getLabel();
			else if(window.isEndLabel())
				currLabel = "NONE";
			exampleString.add(window);
			if(trainWord2Vec)
				vec.addSentence(window.asTokens());
			trainingOutput.add(outcome(currLabel));
			stringOutcomes.add(currLabel);
		}


	}

	public double[] outcome(String label) {
		//always count the none label
		double[] ret = new double[labels.size() + 1];
		//accommodate NONE at beginning
		for(int i = 0; i < labels.size() + 1; i++) {
			ret[i] = 0;
		}
		int idx = labels.indexOf(label);
		if(idx < 0) {
			return new double[] { 1.0,0.0 };
		}
		else 
			return new double[] { 0.0,1.0 };

	}

	public double[][] encode(String input) {
		if(input == null || input.isEmpty())
			return new double[][]{};
		List<String> words = new ArrayList<String>();
		StringTokenizer tokenizer = new StringTokenizer(new InputHomogenization(input).transform());

		List<double[]> ret = new ArrayList<double[]>();
		while(tokenizer.hasMoreTokens()) {
			String token = tokenizer.nextToken();
			//training label
			if(token.charAt(0) == '<' && token.endsWith(">"))
				token = token.toUpperCase();
			words.add(token);
		}

		List<Window> windows = Windows.windows(words,vec.getWindow());

		for(int i = 0; i < windows.size(); i++) {
			ret.add(WindowConverter.asExample(windows.get(i),vec));
		}
		return ret.toArray(new double[][]{});


	}

	

	/* only need to add post trained word vectors */
	private void addWordExamplesAsVectors() {
		if(exampleString != null)
			for(Window training: exampleString) 
				trainingExamples.add(WindowConverter.asExample(training,vec));
		else log.warn("Unable to add word vectors; exampleString is null");
	}



	public Word2Vec getVec() {
		return vec;
	}



	public void train() {
		//train the word vectors
		if(trainWord2Vec)
			vec.train();
		log.info("Built vocab");
		//create the examples based on the word vectors
		addWordExamplesAsVectors();


		log.info("Training " + trainingExamples.size() + " examples");

		for(int i = 0; i < exampleString.size(); i++) {
			Window window = exampleString.get(i);
			log.info("Window " + window.asTokens() + " with label " + stringOutcomes.get(i) + " and focus word " + window.getFocusWord() + " and begin label " + window.isBeginLabel());
		}



		int[] numNodes = {500,500};
		DoubleMatrix inputs = new DoubleMatrix(getInputs());
		DoubleMatrix outputs = new DoubleMatrix(getOutputs());
		CounterMap<Integer,Double> stats = new CounterMap<Integer,Double>();

		for(int i = 0; i < outputs.columns; i++) {
			DoubleMatrix col = outputs.getColumn(i);

			for(int j = 0; j < col.rows; j++) 
				stats.incrementCount(i,col.get(j),1.0);

		}
		stats.normalize();
		log.info(stats.toString());
		if(trainDeepNet) {
			DataSet d = new DataSet(inputs,outputs);

			prop = new Word2VecMultiLayerNetwork.Builder().withWord2Vec(vec)
			.hiddenLayerSizes(numNodes)
			.numberOfInputs(inputs.columns)
			.numberOfOutPuts(outputs.columns)
			.withRng(new MersenneTwister(123))
			.build();


			DataSet train = d;
			prop.trainNetwork(train.getFirst(), train.getSecond(), deepLearningParams);
			initializeViterbi(d);
		}
	

	}

	private void initializeViterbi(DataSet d) {
	
		CounterMap<Integer,Integer> transitionProbabilities = new CounterMap<>();

		
		for(int i = 1; i < d.getSecond().rows; i++) {
			transitionProbabilities.incrementCount(SimpleBlas.iamax(d.getSecond().getRow(i - 1)),SimpleBlas.iamax(d.getSecond().getRow(i)), 1.0);
		}




		DoubleMatrix transitionProbabilities2 = CounterUtil.convert(transitionProbabilities);

		Index labelIndex = new Index();
		labelIndex.add("NONE");
		for(int i = 0 ; i < labels.size(); i++)
			labelIndex.add(labels.get(i));

		//the output single labelStrings are input to the sequence classifier for viterbi
		viterbi = new Viterbi(labelIndex,ViterbiUtil.featureIndexFromLabelIndex(labelIndex),transitionProbabilities2);
	}
	
	public List<String>  predict(String words) {
		words = new InputHomogenization(words).transform();
		List<Window> windows = Windows.windows(words);
		List<String> labels = new ArrayList<String>();

		//individual scores for each window
		DoubleMatrix classified = classify(words);

		List<Datum> datums = viterbi.decode(classified, labels, windows);

		List<String> ret = new ArrayList<String>();
		for(Datum d : datums)
			ret.add(d.guessLabel);
		return ret;
	}

	



	public DoubleMatrix classify(String words) {
		log.info("RUNNING " + words );
		List<Window> windows = Windows.windows(new InputHomogenization(words).transform());
		double[][] ret = new double[windows.size()][labels.size() + 1];

		
		
		for(int i = 0; i < windows.size(); i++) {
			Window window = windows.get(i);
			log.info("PREDICTION FOR " + window);
			DoubleMatrix toPredict = prop.predict(new DoubleMatrix(WindowConverter.asExample(window,vec)));
			ret[i] = toPredict.toArray();
			log.info(toPredict.toString());
		}
		log.info("============================");
		return new DoubleMatrix(ret);
	}

	public static NERClassifier load(InputStream is) throws Exception {
		log.info("LOADING MODEL...");
		ObjectInputStream ois = new ObjectInputStream(is);
		NERClassifier ner = (NERClassifier) ois.readObject();
		return ner;
	}

	public void save(File file) throws  IOException {
		if(file.exists())
			file.delete();
		file.createNewFile();

		ObjectOutputStream ois = new ObjectOutputStream(new BufferedOutputStream(new FileOutputStream(file)));
		ois.writeObject(this);
		ois.flush();
		ois.close();
	}


}
