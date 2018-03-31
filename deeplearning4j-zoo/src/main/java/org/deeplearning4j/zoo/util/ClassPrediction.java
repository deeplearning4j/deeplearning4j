package org.deeplearning4j.zoo.util;

import lombok.AllArgsConstructor;
import lombok.Data;

/**
 * ClassPrediction: a prediction for classification, used with a {@link Labels} class.
 * Holds class number, label description, and the prediction probability.
 *
 * @author saudet
 */
@AllArgsConstructor
@Data
public class ClassPrediction {

    public ClassPrediction(int number, String label, double probability) {
		super();
		this.number = number;
		this.label = label;
		this.probability = probability;
	}

	private int number;
    private String label;
    private double probability;

    @Override
    public String toString() {
        return "ClassPrediction(number=" + number + ",label=" + label + ",probability=" + probability + ")";
    }

	public String getLabel() {
		return label;
	}

	public void setLabel(String label) {
		this.label = label;
	}
}
