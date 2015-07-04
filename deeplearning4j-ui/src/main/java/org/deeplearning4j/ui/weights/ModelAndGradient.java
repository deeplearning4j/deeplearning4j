package org.deeplearning4j.ui.weights;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.annotation.JsonSerialize;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.OutputLayer;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.serializers.GradientSerializer;
import org.deeplearning4j.ui.serializers.ModelSerializer;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.Serializable;
import java.util.Arrays;

/**
 * @author Adam Gibson
 */
public class ModelAndGradient implements Serializable {
    @JsonSerialize(using = GradientSerializer.class, as = Gradient.class)
    private Gradient gradient;
    @JsonSerialize(using = ModelSerializer.class, as= Model.class)
    private Model model;

    public ModelAndGradient() {
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .lossFunction(LossFunctions.LossFunction.MCXENT).optimizationAlgo(OptimizationAlgorithm.ITERATION_GRADIENT_DESCENT)
                .activationFunction("softmax")
                .iterations(10).weightInit(WeightInit.XAVIER)
                .learningRate(1e-1).nIn(4).nOut(3).layer(new org.deeplearning4j.nn.conf.layers.OutputLayer()).build();

        OutputLayer l = LayerFactories.getFactory(conf.getLayer()).create(conf, Arrays.<IterationListener>asList(new ScoreIterationListener(1)),0);
        this.model = l;
        l.setInput(Nd4j.ones(4));
        l.setLabels(Nd4j.ones(3));
        this.gradient = l.gradient();
    }

    public ModelAndGradient(Model model) {
        this.model = model;
        this.gradient = model.gradient();
    }

    @JsonProperty
    public Gradient getGradient() {
        return gradient;
    }

    @JsonProperty
    public void setGradient(Gradient gradient) {
        this.gradient = gradient;
    }

    public Model getModel() {
        return model;
    }

    public void setModel(Model model) {
        this.model = model;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        ModelAndGradient that = (ModelAndGradient) o;

        if (gradient != null ? !gradient.equals(that.gradient) : that.gradient != null) return false;
        return !(model != null ? !model.equals(that.model) : that.model != null);

    }

    @Override
    public int hashCode() {
        int result = gradient != null ? gradient.hashCode() : 0;
        result = 31 * result + (model != null ? model.hashCode() : 0);
        return result;
    }
}
