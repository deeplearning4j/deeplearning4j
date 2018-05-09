package org.deeplearning4j.zoo.util;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

/**
 * Base functionality for helper classes that return label descriptions.
 *
 * @author saudet
 */
public abstract class BaseLabels implements Labels {

    protected ArrayList<String> labels = null;

    /** Override {@link #getLabels()} when using this constructor. */
    protected BaseLabels() throws IOException {
        this.labels = getLabels();
    }

    /**
     * No need to override anything with this constructor.
     *
     * @param textResource name of a resource containing labels as a list in a text file.
     * @throws IOException 
     */
    protected BaseLabels(String textResource) throws IOException {
        this.labels = getLabels(textResource);
    }

    /**
     * Override to return labels when not calling {@link #BaseLabels(String)}.
     */
    protected ArrayList<String> getLabels() throws IOException {
        return null;
    }

    /**
     * Returns labels based on the text file resource.
     */
    protected ArrayList<String> getLabels(String textResource) throws IOException {
        ArrayList<String> labels = new ArrayList<>();
        try (Scanner s = new Scanner(this.getClass().getResourceAsStream(textResource))) {
            while (s.hasNextLine()) {
                labels.add(s.nextLine());
            }
        }
        return labels;
    }

    @Override
    public String getLabel(int n) {
        return labels.get(n);
    }

    @Override
    public List<List<ClassPrediction>> decodePredictions(INDArray predictions, int n) {
        int rows = predictions.size(0);
        int cols = predictions.size(1);
        if (predictions.isColumnVectorOrScalar()) {
            predictions = predictions.ravel();
            rows = predictions.size(0);
            cols = predictions.size(1);
        }
        List<List<ClassPrediction>> descriptions = new ArrayList<>();
        for (int batch = 0; batch < rows; batch++) {
            INDArray result = predictions.getRow(batch);
            result = Nd4j.vstack(Nd4j.linspace(0, cols, cols), result);
            result = Nd4j.sortColumns(result, 1, false);
            List<ClassPrediction> current = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                int label = result.getInt(0, i);
                double prob = result.getDouble(1, i);
                current.add(new ClassPrediction(label, getLabel(label), prob));
            }
            descriptions.add(current);
        }
        return descriptions;
    }

}
