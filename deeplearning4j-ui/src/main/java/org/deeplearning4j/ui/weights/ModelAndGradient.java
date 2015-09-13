package org.deeplearning4j.ui.weights;


import org.deeplearning4j.nn.api.Model;
import org.nd4j.linalg.api.ndarray.INDArray;


import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * @author Adam Gibson
 */

public class ModelAndGradient implements Serializable {
    private Map<String,INDArray> parameters;
    private Map<String,INDArray> gradients;
    private double score;
    private List<Double> scores = new ArrayList<>();
    private String path;


    public ModelAndGradient() {
        parameters = new HashMap<>();
        gradients = new HashMap<>();
    }

    public ModelAndGradient(Model model) {
        this.gradients = model.gradient().gradientForVariable();
        this.parameters = model.paramTable();
        this.score = model.score();
    }




    public double getScore() {
        return score;
    }

    public void setScore(double score) {
        this.score = score;
    }


    public Map<String, INDArray> getParameters() {
        return parameters;
    }

    public void setParameters(Map<String, INDArray> parameters) {
        this.parameters = parameters;
    }


    public Map<String, INDArray> getGradients() {
        return gradients;
    }

    public void setGradients(Map<String, INDArray> gradients) {
        this.gradients = gradients;
    }

    public void setScores(List<Double> scores){
        this.scores = scores;
    }

    public void setPath(String path){
        this.path = path;
    }

    public String getPath(){
        return path;
    }

    public List<Double> getScores(){
        return scores;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        ModelAndGradient that = (ModelAndGradient) o;

        if (Double.compare(that.score, score) != 0) return false;
        if (parameters != null ? !parameters.equals(that.parameters) : that.parameters != null) return false;
        return !(gradients != null ? !gradients.equals(that.gradients) : that.gradients != null);

    }

    @Override
    public int hashCode() {
        int result;
        long temp;
        result = parameters != null ? parameters.hashCode() : 0;
        result = 31 * result + (gradients != null ? gradients.hashCode() : 0);
        temp = Double.doubleToLongBits(score);
        result = 31 * result + (int) (temp ^ (temp >>> 32));
        return result;
    }
}
