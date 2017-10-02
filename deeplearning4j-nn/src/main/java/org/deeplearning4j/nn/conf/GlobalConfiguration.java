package org.deeplearning4j.nn.conf;

import lombok.Builder;
import lombok.Data;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.api.layers.LayerConstraint;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.dropout.IDropout;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.stepfunctions.StepFunction;
import org.deeplearning4j.nn.conf.weightnoise.IWeightNoise;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationSigmoid;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.Sgd;

import java.lang.reflect.Field;
import java.util.List;

@Data
public class GlobalConfiguration {

    protected IActivation activationFn = new ActivationSigmoid();
    protected WeightInit weightInit = WeightInit.XAVIER;
    protected Double biasInit;
    protected Distribution dist = null;
    protected Double l1 = 0.0;
    protected Double l2 = 0.0;
    protected Double l1Bias;
    protected Double l2Bias;
    protected IDropout dropOut;
    protected IWeightNoise weightNoise;
    protected IUpdater updater = new Sgd();
    protected IUpdater biasUpdater;
    protected Layer layer;
    protected Boolean miniBatch = true;
    protected Integer maxNumLineSearchIterations = 5;
    protected Long seed = System.currentTimeMillis();
    protected OptimizationAlgorithm optimizationAlgo = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT;
    protected StepFunction stepFunction;
    protected Boolean minimize = true;
    protected GradientNormalization gradientNormalization = GradientNormalization.None;
    protected Double gradientNormalizationThreshold = 1.0;
    protected Boolean pretrain = false;
    protected List<LayerConstraint> allParamConstraints;
    protected List<LayerConstraint> weightConstraints;
    protected List<LayerConstraint> biasConstraints;

    protected WorkspaceMode trainingWorkspaceMode = WorkspaceMode.NONE;
    protected WorkspaceMode inferenceWorkspaceMode = WorkspaceMode.SEPARATE;
    protected CacheMode cacheMode = CacheMode.NONE;

    protected ConvolutionMode convolutionMode = ConvolutionMode.Truncate;


    public GlobalConfiguration(boolean withDefaults){
        if(!withDefaults){
            clear();
        }
    }



    public void clear(){
        Field[] fields = GlobalConfiguration.class.getDeclaredFields();
        for(Field f : fields){
            f.setAccessible(true);
            try{
                f.set(this, null);
            } catch (IllegalAccessException e){
                throw new RuntimeException(e);  //Should never happen
            }
        }
    }

    @Override
    public GlobalConfiguration clone(){
        throw new UnsupportedOperationException("Not yet implemneted");
    }


}
