package com.ccc.sendalyzeit.deeplearning.eval;


import org.jblas.DoubleMatrix;
import org.jblas.SimpleBlas;

import com.ccc.sendalyzeit.deeplearning.berkeley.Counter;

/**
 * Evaluation metrics: precision, recall, f1
 * @author Adam Gibson
 *
 */
public class Evaluation {

	private double truePositives;
	private Counter<Integer> falsePositives = new Counter<Integer>();
	private double falseNegatives;
	private ConfusionMatrix<Integer> confusion = new ConfusionMatrix<Integer>();

	/**
	 * Collects statistics on the real outcomes vs the 
	 * guesses. This is for logistic outcome matrices such that the 
	 * 
	 * Note that an IllegalArgumentException is thrown if the two passed in
	 * matrices aren't the same length.
	 * @param realOutcomes the real outcomes (usually binary)
	 * @param guesses the guesses (usually a probability vector)
	 */
	public void eval(DoubleMatrix realOutcomes,DoubleMatrix guesses) {
		if(realOutcomes.length != guesses.length)
			throw new IllegalArgumentException("Unable to evaluate. Outcome matrices not same length");
		for(int i = 0; i < realOutcomes.rows; i++) {
			DoubleMatrix currRow = realOutcomes.getRow(i);
			DoubleMatrix guessRow = guesses.getRow(i);

			int currMax = SimpleBlas.iamax(currRow);
			int guessMax = SimpleBlas.iamax(guessRow);

			addToConfusion(currMax,guessMax);

			if(currMax == guessMax)
				incrementTruePositives();
			else {
				incrementFalseNegatives();
				incrementFalsePositives(guessMax);
			}
		}
	}

	
	public String stats() {
		StringBuilder builder = new StringBuilder()
		.append("\n").append(confusion.toString()).append("\n");
		
		
		builder.append("\n==========================F1 Scores========================================");
		
		for(Integer label : confusion.getClasses()) {
			builder.append("\nLABEL " + label + " f1 " + f1(label));
		}
		builder.append("\n===========================================================================");
		return builder.toString();
	}
	
	/**
	 * Adds to the confusion matrix
	 * @param real the actual guess
	 * @param guess the system guess
	 */
	public void addToConfusion(int real,int guess) {
		confusion.add(real, guess);
	}

	/**
	 * Returns the number of times the given label
	 * has actually occurred
	 * @param i the label
	 * @return the number of times the label
	 * actually occurred
	 */
	public int classCount(int i) {
		return confusion.getActualTotal(i);
	}
	
	/**
	 * Returns the number of times a given label was predicted 
	 * @param label the label to get
	 * @return the number of times the given label was predicted
	 */
	public int numtimesPredicted(int label) {
		return confusion.getPredictedTotal(label);
	}
	
	/**
	 * Gets the number of times the 
	 * given class was predicted for the 
	 * given predicted label
	 * @param actual 
	 * @param predicted
	 * @return
	 */
	public int numTimesPredicted(int actual,int predicted) {
		return confusion.getCount(actual, predicted);
	}
	
	/**
	 * Calculate f1 score for a given class
	 * @param i the label to calculate f1 for
	 * @return the f1 score for the given label
	 */
	public double f1(int i) {
		double precision = precision(i);
		double recall = recall();
		return 2.0 * ((precision * recall / (precision + recall)));
	}
	
	/**
	 * Returns the recall for the outcomes
	 * @return the recall for the outcomes
	 */
	public double recall() {
		return truePositives / (truePositives + falseNegatives);
	}
	
	/**
	 * Returns the precision for a given label
 	 * @param i the label
	 * @return the precision for the label
	 */
	public double precision(int i) {
		return truePositives / (truePositives + falsePositives.getCount(i));
	}


	public void incrementTruePositives() {
		truePositives++;
	}

	public void incrementFalseNegatives() {
		falseNegatives++;
	}

	public void incrementFalsePositives(int i) {
		falsePositives.incrementCount(i, 1.0);
	}


}
