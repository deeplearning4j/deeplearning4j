package com.ccc.deeplearning.word2vec.viterbi;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;

import org.jblas.DoubleMatrix;
import org.jblas.SimpleBlas;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.ccc.deeplearning.nn.Persistable;
import com.ccc.deeplearning.word2vec.util.Window;

/**
 * Viterbi implementation
 * @author Adam Gibson
 *
 */
public class Viterbi implements Persistable {


	private static final long serialVersionUID = 3254568492760166461L;
	private Index labelIndex;
	private Index featureIndex;
	private DoubleMatrix weights;
	private static Logger log = LoggerFactory.getLogger(Viterbi.class);
	
	public Viterbi(Index labelIndex, Index featureIndex, DoubleMatrix weights) {
		this.labelIndex = labelIndex;
		this.featureIndex = featureIndex;
		this.weights = weights;
	}

	private Viterbi() {}


	public List<Datum> decode(DoubleMatrix classified,List<String> labels,List<Window> windows) {

		List<String> previousLabels = new ArrayList<String>();

		//discretize to max probability for individual outcomes
		for(DoubleMatrix row : classified.rowsAsList()) {
			int idx = SimpleBlas.iamax(row);
			if(idx < 1)
				labels.add("NONE");
			else
				labels.add(labels.get(idx));
		}
		previousLabels.add("NONE");

		//add the previous labelStrings for each window
		for(int i = 1; i < labels.size(); i++) 
			previousLabels.add(labels.get(i - 1));

		//assigns features as outcomes
		List<Datum> datums = Datum.datums(windows, labels, previousLabels,ViterbiUtil.toFeatures(classified));
		decode(datums, ViterbiUtil.previousLabelDatums(datums));
		return datums;
	}


	/**
	 * Classify the given sequence
	 * @param data the data to classify
	 * @param dataWithMultiplePrevLabels the data with 
	 * sequences
	 */
	public void decode(List<Datum> data, List<Datum> dataWithMultiplePrevLabels) {
		// load words from the data
		List<String> words = new ArrayList<String>();
		for (Datum datum : data) {
			words.add(datum.word);
			if(datum.features.size() != numLabels())
				throw new IllegalArgumentException("Datum for word " + datum.word + " does not have the right number of features. These must be the label equivalents.");
		}

		int[][] backpointers = new int[data.size()][numLabels()];
		DoubleMatrix scores = new DoubleMatrix(data.size(),numLabels());

		int prevLabel = labelIndex.indexOf(data.get(0).previousLabel);
		DoubleMatrix localScores = computeScores(data.get(0).features);

		int position = 0;
		for (int currLabel = 0; currLabel < localScores.length; currLabel++) {
			backpointers[position][currLabel] = prevLabel;
			scores.put(position,currLabel,localScores.get(currLabel));
		}

		// for each position in data
		for (position = 1; position< data.size(); position++) {
			// equivalent position in dataWithMultiplePrevLabels
			int i = position * numLabels() - 1; 

			// for each previous label 
			for (int j = 0; j < numLabels(); j++) {
				int datumToEval = i + j;
				if(datumToEval >=  dataWithMultiplePrevLabels.size())
					break;
				Datum datum = dataWithMultiplePrevLabels.get(i + j);


				String previousLabel = datum.previousLabel;

				if(previousLabel == null)
					throw new IllegalStateException("Datum previous word can NEVER be null");


				prevLabel = labelIndex.indexOf(previousLabel);

				localScores = computeScores(datum.features);
				for (int currLabel = 0; currLabel < localScores.length; currLabel++) {
					double score = localScores.get(currLabel) + scores.get(position - 1,prevLabel);
					if (prevLabel == 0 || score > scores.get(position,currLabel)) {
						backpointers[position][currLabel] = prevLabel;
						scores.put(position,currLabel,score);
					}
				}
			}
		}

		int bestLabel = 0;
		double bestScore = scores.get(data.size() - 1,0);

		for (int label = 1; label < scores.getRow(data.size() - 1).length; label++) {
			if (scores.get(data.size() - 1,label) > bestScore) {
				bestLabel = label;
				bestScore = scores.get(data.size() - 1,label);
			}
		}

		for (position = data.size() - 1; position >= 0; position--) {
			Datum datum = data.get(position);
			datum.guessLabel = (String) labelIndex.get(bestLabel);
			bestLabel = backpointers[position][bestLabel];
		}

	}

	private DoubleMatrix computeScores(List<String> features) {

		DoubleMatrix scores = new DoubleMatrix(numLabels());

		for (Object feature : features) {
			int f = featureIndex.indexOf(feature);
			if (f < 0) 
				continue;

			for (int i = 0; i < scores.length; i++) 
				scores.put(i,weights.get(i,f));

		}

		return scores;
	}

	private int numLabels() {
		return labelIndex.size();
	}

	
	
	public static Viterbi load(String path) throws IOException {
		Viterbi v = new Viterbi();
		log.info("Loading viterbi model...");
		v.load(new BufferedInputStream(new FileInputStream(new File(path))));
		return v;
	}
	
	@Override
	public void write(OutputStream os) {
		try {
			ObjectOutputStream os2 = new ObjectOutputStream(os);
			os2.writeObject(this);
		} catch (IOException e) {
			throw new RuntimeException(e);
		}

	}

	@Override
	public void load(InputStream is) {
		try {
			ObjectInputStream ois = new ObjectInputStream(is);
			Viterbi v = (Viterbi) ois.readObject();
			this.featureIndex = v.featureIndex;
			this.labelIndex = v.labelIndex;
		} catch (Exception e) {
			throw new RuntimeException(e);
		}


	}

}