/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.arbiter.eval;

import java.util.Set;

import org.arbiter.berkeley.Counter;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Evaluation metrics: precision, recall, f1
 * @author Adam Gibson
 *
 */
public class Evaluation {

	private Counter<Integer> truePositives = new Counter<>();
	private Counter<Integer> falsePositives = new Counter<>();
	private Counter<Integer> trueNegative = new Counter<>();
	private Counter<Integer> falseNegatives = new Counter<>();
	private ConfusionMatrix<Integer> confusion = new ConfusionMatrix<>();

	/**
	 * Collects statistics on the real outcomes vs the 
	 * guesses. This is for logistic outcome matrices such that the 
	 * 
	 * Note that an IllegalArgumentException is thrown if the two passed in
	 * matrices aren't the same length.
	 * @param realOutcomes the real outcomes (usually binary)
	 * @param guesses the guesses (usually a probability vector)
	 */
	public void eval(INDArray realOutcomes,INDArray guesses) {
		if(realOutcomes.length() != guesses.length())
			throw new IllegalArgumentException("Unable to evaluate. Outcome matrices not same length");
		for(int i = 0; i < realOutcomes.rows(); i++) {
			INDArray currRow = realOutcomes.getRow(i);
			INDArray guessRow = guesses.getRow(i);

			int currMax;
            {
                double max = currRow.getDouble(0);
                currMax = 0;
                for (int col = 1; col < currRow.columns(); col++) {
                    if (currRow.getDouble(col) > max) {
                        max = currRow.getDouble(col);
                        currMax = col;
                    }
                }
            }
            int guessMax;
            {
                double max = guessRow.getDouble(0);
                guessMax = 0;
                for (int col = 1; col < guessRow.columns(); col++) {
                    if (guessRow.getDouble(col) > max) {
                        max = guessRow.getDouble(col);
                        guessMax = col;
                    }
                }
            }

			addToConfusion(currMax,guessMax);

			if(currMax == guessMax) {
				incrementTruePositives(guessMax);
				for(Integer clazz : confusion.getClasses()) {
					if(clazz != guessMax)
						trueNegative.incrementCount(clazz, 1.0);
				}
			}
			else {
				incrementFalseNegatives(currMax);
				incrementFalsePositives(guessMax);
			}
			
			
			
		}
	}



	public String stats() {
		StringBuilder builder = new StringBuilder()
		.append("\n");
		Set<Integer> classes = confusion.getClasses();
		for(Integer clazz : classes) {
			for(Integer clazz2 : classes) {
				int count = confusion.getCount(clazz, clazz2);
				if(count != 0)
					builder.append("\nActual Class " + clazz + " was predicted with Predicted " + clazz2 + " with count " + count  + " times\n");
			}
		}
		builder.append("\n==========================F1 Scores========================================");
		builder.append("\n " + f1());
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
	 * @param label the label to getFromOrigin
	 * @return the number of times the given label was predicted
	 */
	public int numtimesPredicted(int label) {
		return confusion.getPredictedTotal(label);
	}

	/**
	 * Gets the number of times the 
	 * given class was predicted for the 
	 * given predicted label
	 * @param actual actual outcome
	 * @param predicted predicted outcome
	 * @return number of time predicted
	 */
	public int numTimesPredicted(int actual,int predicted) {
		return confusion.getCount(actual, predicted);
	}

	/**
	 * Total precision based on guesses so far
	 * @return the total precision based on guesses so far
	 * 
	 */
	public double precision() {
		double prec = 0.0;
		for(Integer i : confusion.getClasses()) {
			prec += precision(i);
		}
		return prec / (double) confusion.getClasses().size();
	}


	/**
	 * True negatives: correctly rejected
	 * @return the total true negatives so far
	 */
	public double trueNegatives() {
		return trueNegative.totalCount();
	}
	
	/**
	 * False positive: wrong guess
	 * @return the count of the false positives
	 */
	public double falsePositive() {
		return falsePositives.totalCount();
	}
	
	/**
	 * Total negatives true negatives + falseNegatives
	 * @return the overall negative count
	 */
	public double negative() {
		return trueNegatives() + falseNegatives.totalCount();
		
	}
	
	/**
	 * Returns all of the positive guesses:
	 * true positive + false negative
	 * @return
	 */
	public double positive() {
		return truePositives.totalCount() + falseNegatives.totalCount();
	}
	
	/**
	 * Accuracy: 
	 * TP + TN / (P + N)
	 * @return the accuracy of the guesses so far
	 */
	public double accuracy() {
		return (this.truePositives.totalCount() + trueNegatives()) / (positive() + negative());
	}
	
	/**
	 * TP: true positive
	 * FP: False Positive
	 * FN: False Negative
	 * F1 score: 2 * TP / (2TP + FP + FN)
	 * @return the f1 score or harmonic mean based on current guesses
	 */
	public double f1() {
		double precision = precision();
		double recall = recall();
		if(precision == 0 || recall == 0)
			return 0;
		return 2.0 * ((precision * recall / (precision + recall)));
	}

	/**
	 * Calculate f1 score for a given class
	 * @param i the label to calculate f1 for
	 * @return the f1 score for the given label
	 */
	public double f1(int i) {
		double precision = precision(i);
		double recall = recall();
		if(precision == 0 || recall == 0)
			return 0;
		return 2.0 * ((precision * recall / (precision + recall)));
	}

	/**
	 * Returns the recall for the outcomes
	 * @return the recall for the outcomes
	 */
	public double recall() {
        double r = 0.0;
        for(Integer i : confusion.getClasses()) {
            r += recall(i);
        }
        return r / (double) confusion.getClasses().size();
	}

    public double recall(int i) {
        if(truePositives.getCount(i) == 0)
            return 0;
        return truePositives.getCount(i) / (truePositives.getCount(i) + falseNegatives.getCount(i));
    }

	/**
	 * Returns the precision for a given label
	 * @param i the label
	 * @return the precision for the label
	 */
	public double precision(int i) {
		if(truePositives.getCount(i) == 0)
			return 0;
		return truePositives.getCount(i) / (truePositives.getCount(i) + falsePositives.getCount(i));
	}


	public void incrementTruePositives(int i) {
		truePositives.incrementCount(i, 1.0);
	}

	public void incrementFalseNegatives(int i) {
		falseNegatives.incrementCount(i, 1.0);
	}

	public void incrementFalsePositives(int i) {
		falsePositives.incrementCount(i, 1.0);
	}


}
