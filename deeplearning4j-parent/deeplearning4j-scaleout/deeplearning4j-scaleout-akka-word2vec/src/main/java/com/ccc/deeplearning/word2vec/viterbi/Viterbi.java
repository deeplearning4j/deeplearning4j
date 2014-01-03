package com.ccc.deeplearning.word2vec.viterbi;

import java.io.Serializable;
import java.util.*;

import org.jblas.DoubleMatrix;

/**
 * Viterbi implementation
 * @author Adam Gibson
 *
 */
public class Viterbi implements Serializable {

	
	private static final long serialVersionUID = 3254568492760166461L;
	private  final Index labelIndex;
	private  final Index featureIndex;
	private final DoubleMatrix weights;

	public Viterbi(Index labelIndex, Index featureIndex, DoubleMatrix weights) {
		this.labelIndex = labelIndex;
		this.featureIndex = featureIndex;
		this.weights = weights;
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

}