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

    private int number;
    private String label;
    private double probability;

    @Override
    public String toString() {
        return "ClassPrediction(number=" + number + ",label=" + label + ",probability=" + probability + ")";
    }
}
