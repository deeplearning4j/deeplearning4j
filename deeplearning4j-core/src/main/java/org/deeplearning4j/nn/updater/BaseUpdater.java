package org.deeplearning4j.nn.updater;

import com.google.common.base.Function;
import org.apache.commons.math3.util.FastMath;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.LearningRatePolicy;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.updater.aggregate.UpdaterAggregator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.accum.Norm2;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.AbsValueGreaterThan;
import org.nd4j.linalg.indexing.conditions.Condition;
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.linalg.learning.GradientUpdaterAggregator;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * @author Adam Gibson
 */
public abstract class BaseUpdater implements Updater {
    protected Map<String, GradientUpdater> updaterForVariable = new HashMap<>();


    @Override
    public void update(Layer layer, Gradient gradient, int iteration, int miniBatchSize) {
        String paramName;
        INDArray paramVal, gradient2;
        GradientUpdater updater;

        preApply(layer, gradient, iteration);
        for (Map.Entry<String, INDArray> gradientPair : gradient.gradientForVariable().entrySet()) {
            paramName = gradientPair.getKey();
            paramVal = gradientPair.getValue();
            LearningRatePolicy decay = layer.conf().getLearningRatePolicy();
            if (decay != LearningRatePolicy.None || layer.conf().getLayer().getUpdater() == org.deeplearning4j.nn.conf.Updater.NESTEROVS)
                applyLrDecayPolicy(decay, layer, iteration, paramName);
            updater = init(paramName, paramVal, layer);
            gradient2 = updater.getGradient(paramVal, iteration);
            postApply(layer, gradient2, paramName, miniBatchSize);
            gradient.setGradientFor(paramName, gradient2);
        }
    }

    /**
     * Apply the regularization
     *
     * @param layer
     * @param gradient
     * @param param
     */
    public void postApply(Layer layer, INDArray gradient, String param, int miniBatchSize) {
        NeuralNetConfiguration conf = layer.conf();
        INDArray params = layer.getParam(param);
        if (conf.isUseRegularization() && conf.getL2ByParam(param) > 0)
            gradient.addi(params.mul(conf.getL2ByParam(param)));    //dC/dw = dC0/dw + lambda/n * w where C0 is pre-l2 cost function
        if (conf.isUseRegularization() && conf.getL1ByParam(param) > 0)
            gradient.addi(Transforms.sign(params).muli(conf.getL1ByParam(param)));
        if (conf.isMiniBatch())
            gradient.divi(miniBatchSize);

    }

    /**
     *  Update momentum if schedule exist
     */
    public void applyMomentumDecayPolicy(Layer layer, int iteration, String variable){
        NeuralNetConfiguration conf = layer.conf();
        if (conf.getLayer().getMomentumSchedule().containsKey(iteration)) {
            conf.getLayer().setMomentum(conf.getLayer().getMomentumSchedule().get(iteration));
            if(updaterForVariable.get(variable) != null)
                updaterForVariable.get(variable).update(conf.getLearningRateByParam(variable), conf.getLayer().getMomentumSchedule().get(iteration));
        }
    }

    /**
     *  Update learning rate based on policy
     */
    public void applyLrDecayPolicy(LearningRatePolicy decay, Layer layer, int iteration, String variable){
        NeuralNetConfiguration conf = layer.conf();
        double decayRate = layer.conf().getLrPolicyDecayRate();
        double lr = conf.getLearningRateByParam(variable);
        switch(decay){
            case Exponential:
                conf.setLearningRateByParam(variable, lr * Math.pow(decayRate, iteration));
                break;
            case Inverse:
                conf.setLearningRateByParam(variable, lr / Math.pow((1+decayRate * iteration), conf.getLrPolicyPower()));
                break;
            case Step:
                conf.setLearningRateByParam(variable, lr * Math.pow(decayRate, Math.floor(iteration/conf.getLrPolicySteps())));
                break;
            case Poly:
                conf.setLearningRateByParam(variable, lr * Math.pow((1 - (iteration * 1.0)/conf.getNumIterations()), conf.getLrPolicyPower()));
                break;
            case Sigmoid:
                conf.setLearningRateByParam(variable, lr / (1 + Math.exp(-decayRate * (iteration - conf.getLrPolicySteps()))));
                break;
            case Schedule:
                if (conf.getLayer().getLearningRateSchedule().containsKey(iteration))
                    conf.setLearningRateByParam(variable, conf.getLayer().getLearningRateSchedule().get(iteration));
                break;
        }
        if(layer.conf().getLayer().getUpdater() == org.deeplearning4j.nn.conf.Updater.NESTEROVS)
            applyMomentumDecayPolicy(layer, iteration, variable);
        else if(updaterForVariable.get(variable) != null)
            updaterForVariable.get(variable).update(conf.getLearningRateByParam(variable));
    }

    /**
     *  Apply gradient normalization: scale based on L2, clipping etc.
     *  RenormalizeL2PerLayer: divide all layer gradients by L2 to rescale
     *  RenormalizeL2PerParamType: divide each parameter type gradient in a layer by L2 to rescale
     *  ClipElementWiseAbsoluteValue: clip gradients per-element
     *  ClipL2PerLayer: same as RenormalizeL2PerLayer but limited by gradient L2 norm for the layer meeting a threshold
     *  ClipL2PerParamType: same as RenormalizeL2PerParamType but limited by gradient L2 norm for each parameter type in a layer meeting a threshold
     */
    public void preApply(Layer layer, Gradient gradient, int iteration) {

        GradientNormalization normalization = layer.conf().getLayer().getGradientNormalization();
        if (normalization == null || normalization == GradientNormalization.None) return;  //no op

        final double threshold = layer.conf().getLayer().getGradientNormalizationThreshold();

        switch (normalization) {
            case RenormalizeL2PerLayer:
                double sumSquares = 0.0;
                for (INDArray g : gradient.gradientForVariable().values()) {
                    double l2 = g.norm2Number().doubleValue();
                    //l2 norm: sqrt(sum_i g_i^2)
                    sumSquares += l2*l2;
                }
                double layerL2 = FastMath.sqrt(sumSquares);
                for (INDArray g : gradient.gradientForVariable().values()) {
                    g.divi(layerL2);
                }
                break;
            case RenormalizeL2PerParamType:
                for (INDArray g : gradient.gradientForVariable().values()) {
                    double l2 = Nd4j.getExecutioner().execAndReturn(new Norm2(g)).getFinalResult().doubleValue();
                    g.divi(l2);
                }
                break;
            case ClipElementWiseAbsoluteValue:
                Condition absValueCondition = new AbsValueGreaterThan(threshold);
                Function<Number,Number> clipFn = new Function<Number, Number>() {
                    @Override
                    public Number apply(Number number) {
                        return (number.doubleValue() > threshold ? threshold : -threshold);
                    }
                };

                for( INDArray g : gradient.gradientForVariable().values()){
                    BooleanIndexing.applyWhere(g, absValueCondition, clipFn);
                }
                break;
            case ClipL2PerLayer:
                double sumSquares2 = 0.0;
                for (INDArray g : gradient.gradientForVariable().values()) {
                    double l2 = Nd4j.getExecutioner().execAndReturn(new Norm2(g)).getFinalResult().doubleValue();
                    //l2 norm: sqrt(sum_i g_i^2)
                    sumSquares2 += l2*l2;
                }
                double layerL22 = FastMath.sqrt(sumSquares2);
                if(layerL22 > threshold ){
                    double scalingFactor = threshold / layerL22;    // g = g / l2 * threshold ->
                    for(INDArray g : gradient.gradientForVariable().values()){
                        g.muli(scalingFactor);
                    }
                }
                break;
            case ClipL2PerParamType:
                for (INDArray g : gradient.gradientForVariable().values()) {
                    double l2 = g.norm2Number().doubleValue();
                    if(l2 > threshold){
                        double scalingFactor = l2 / threshold;
                        g.divi(scalingFactor);
                    }
                }
                break;
            default:
                throw new RuntimeException("Unknown (or not implemented) gradient normalization strategy: " + normalization);
        }
    }


    public abstract void init();

    public abstract GradientUpdater init(String variable, INDArray gradient, Layer layer);

    protected static abstract class UpdaterAggregatorImpl implements UpdaterAggregator {
        protected Map<String,GradientUpdaterAggregator> aggregatorMap = new LinkedHashMap<>();

        @Override
        public void aggregate(Updater updater) {
            BaseUpdater bu = (BaseUpdater)updater;
            for(String s : bu.updaterForVariable.keySet() ) {
                GradientUpdaterAggregator ag = aggregatorMap.get(s);
                GradientUpdater guToAdd = bu.updaterForVariable.get(s);

                if(ag == null){
                    ag = guToAdd.getAggregator(true);
                    aggregatorMap.put(s,ag);
                    continue;
                }

                ag.aggregate(guToAdd);
            }
        }

        @Override
        public void merge(UpdaterAggregator aggregator) {
            UpdaterAggregatorImpl ag = (UpdaterAggregatorImpl)aggregator;
            if(aggregatorMap == null){
                aggregatorMap = ag.aggregatorMap;
            } else {
                if(ag.aggregatorMap == null) return;
                for(String s : ag.aggregatorMap.keySet() ) {
                    GradientUpdaterAggregator first = aggregatorMap.get(s);
                    GradientUpdaterAggregator second = ag.aggregatorMap.get(s);
                    first.combine(second);
                }
            }
        }

        protected Updater setUpdaterState(BaseUpdater updater){
            updater.updaterForVariable = new LinkedHashMap<>();
            for(String s : this.aggregatorMap.keySet() ){
                updater.updaterForVariable.put(s,this.aggregatorMap.get(s).getUpdater());
            }
            return updater;
        }
    }

    @Override
    public boolean equals(Object other){
        if(!(other instanceof BaseUpdater)) return false;
        return updaterForVariable.equals(((BaseUpdater)other).updaterForVariable);
    }

    @Override
    public Updater clone(){
        Map<String,GradientUpdater> newMap = new HashMap<>();
        for( String s : updaterForVariable.keySet()){
            newMap.put(s, updaterForVariable.get(s).getAggregator(true).getUpdater());
        }

        BaseUpdater updater;
        try{
            updater = this.getClass().getConstructor().newInstance();
        }catch (Exception e){
            throw new RuntimeException(e);
        }
        updater.updaterForVariable = newMap;
        return updater;
    }
}
