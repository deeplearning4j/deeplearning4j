package org.deeplearning4j.nn.api.gradients;

import org.deeplearning4j.nn.api.activations.ActivationsFactory;
import org.deeplearning4j.nn.gradient.Gradient;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.concurrent.atomic.AtomicLong;

public class GradientsFactory {

    private static final int DEFAULT_CACHE_SIZE = 256;
    private static final GradientsFactory INSTANCE = new GradientsFactory();
    public static GradientsFactory getInstance(){
        return INSTANCE;
    }


    private GradientsSingle[] singles;
    private GradientsPair[] pairs;
    private GradientsTriple[] triples;
    private Gradients[] tuples;

    private AtomicLong singlesStart = new AtomicLong(-1);
    private AtomicLong singlesEnd = new AtomicLong(-1);
    private AtomicLong pairsStart = new AtomicLong(-1);
    private AtomicLong pairsEnd = new AtomicLong(-1);
    private AtomicLong triplesStart = new AtomicLong(-1);
    private AtomicLong triplesEnd = new AtomicLong(-1);


    private GradientsFactory(){
        this(DEFAULT_CACHE_SIZE);
    }

    private GradientsFactory(int cacheSize){
        singles = new GradientsSingle[cacheSize];
        pairs = new GradientsPair[cacheSize];
        triples = new GradientsTriple[cacheSize];
        tuples = new Gradients[cacheSize];
    }

    public void release(Gradients gradients){
        if(gradients == null) return;
        gradients.clear();

        if(gradients instanceof GradientsSingle){
            ActivationsFactory.set(gradients, singles, singlesStart, singlesEnd);
        } else if(gradients instanceof GradientsPair){
            ActivationsFactory.set(gradients, pairs, pairsStart, pairsEnd);
        } else if(gradients instanceof GradientsTriple){
            ActivationsFactory.set(gradients, triples, triplesStart, triplesEnd);
        }
    }

    public Gradients create(INDArray actGrad){
        return create(actGrad, null);
    }

    public Gradients create(INDArray actGrad, Gradient paramGradient){
        //First: determine if any cached value is available
        GradientsSingle single = ActivationsFactory.get(singles, singlesStart, singlesEnd);
        if(single != null){
            single.set(0, actGrad);
            single.setParameterGradients(paramGradient);
        }
        return new GradientsSingle(actGrad, paramGradient);
    }

    public Gradients createPair(INDArray actGrad1, INDArray actGrad2, Gradient paramGrad) {
        //First: determine if any cached value is available
        GradientsPair pair = ActivationsFactory.get(pairs, pairsStart, pairsEnd);
        if(pair != null){
            pair.set(0, actGrad1);
            pair.set(1, actGrad2);
            pair.setParameterGradients(paramGrad);
        }
        return new GradientsPair(actGrad1, actGrad2, paramGrad);
    }

    public Gradients createTriple(INDArray actGrad1, INDArray actGrad2, INDArray actGrad3, Gradient paramGrad) {
        //First: determine if any cached value is available
        GradientsTriple triple = ActivationsFactory.get(triples, triplesStart, triplesEnd);
        if(triple != null){
            triple.set(0, actGrad1);
            triple.set(1, actGrad2);
            triple.set(2, actGrad3);
            triple.setParameterGradients(paramGrad);
        }
        return new GradientsTriple(actGrad1, actGrad2, actGrad3, paramGrad);
    }

    public Gradients create(Gradient paramGrad, INDArray... actGrad){
        return new GradientsTuple(actGrad, paramGrad);
    }
}
