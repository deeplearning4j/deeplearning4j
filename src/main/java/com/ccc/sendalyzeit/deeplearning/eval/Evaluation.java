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
      * @param guesses the guesses (usually a probaility vector)
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
     
     /**
      * Adds to the confusion matrix
      * @param real the actual guess
      * @param guess the system guess
      */
     public void addToConfusion(int real,int guess) {
    	 confusion.add(real, guess);
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
     
     public double recall() {
    	 return truePositives / (truePositives + falseNegatives);
     }
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
