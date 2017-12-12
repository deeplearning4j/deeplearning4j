package org.deeplearning4j.nn.layers.recurrent;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.params.SimpleRnnParamInitializer;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastCopyOp;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

public class SimpleRnn extends BaseRecurrentLayer<org.deeplearning4j.nn.conf.layers.recurrent.SimpleRnn> {
    public static final String STATE_KEY_PREV_ACTIVATION = "prevAct";

    public SimpleRnn(NeuralNetConfiguration conf) {
        super(conf);
    }

    @Override
    public INDArray rnnTimeStep(INDArray input) {
        INDArray last = stateMap.get(STATE_KEY_PREV_ACTIVATION);
        INDArray out = activateHelper(input, last, false, false).getFirst();
        try(MemoryWorkspace ws = Nd4j.getWorkspaceManager().scopeOutOfWorkspaces()){
            stateMap.put(STATE_KEY_PREV_ACTIVATION, out.get(all(), all(), point(out.size(2)-1)));
        }
        return out;
    }

    @Override
    public INDArray rnnActivateUsingStoredState(INDArray input, boolean training, boolean storeLastForTBPTT) {
        INDArray last = tBpttStateMap.get(STATE_KEY_PREV_ACTIVATION);
        INDArray out = activateHelper(input, last, training, false).getFirst();
        if(storeLastForTBPTT){
            try(MemoryWorkspace ws = Nd4j.getWorkspaceManager().scopeOutOfWorkspaces()){
                tBpttStateMap.put(STATE_KEY_PREV_ACTIVATION, out.get(all(), all(), point(out.size(2)-1)));
            }
        }
        return out;
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon) {
        return tbpttBackpropGradient(epsilon, -1);
    }

    @Override
    public Pair<Gradient, INDArray> tbpttBackpropGradient(INDArray epsilon, int tbpttBackLength) {
        //First: Do forward pass to get gate activations and Zs
        Pair<INDArray,INDArray> p = activateHelper(input, null, true, true);

        INDArray w = getParam(SimpleRnnParamInitializer.WEIGHT_KEY);
        INDArray rw = getParam(SimpleRnnParamInitializer.RECURRENT_WEIGHT_KEY);
        INDArray b = getParam(SimpleRnnParamInitializer.BIAS_KEY);

        INDArray wg = gradientViews.get(SimpleRnnParamInitializer.WEIGHT_KEY);
        INDArray rwg = gradientViews.get(SimpleRnnParamInitializer.WEIGHT_KEY);
        INDArray bg = gradientViews.get(SimpleRnnParamInitializer.WEIGHT_KEY);


        int nOut = layerConf().getNOut();
        int tsLength = input.size(1);

        INDArray epsOut = Nd4j.createUninitialized(input.shape(), 'f');
        IActivation a = layerConf().getActivationFn();

        INDArray dldzNext = null;
        for( int i = tsLength; i>= 0; i--){
            INDArray dldaCurrent = epsilon.get(all(), all(), point(i));
            INDArray zCurrent = p.getSecond().get(all(), all(), point(i));
            INDArray inCurrent = input.get(all(), all(), point(i));
            INDArray epsOutCurrent = epsOut.get(all(), all(), point(i));
            INDArray aLast;
            if(i > 0){
                aLast = p.getFirst().get(all(), all(), point(i-1));
            } else {
                aLast = null;
            }

            if(dldzNext != null){
                dldaCurrent.addi(dldzNext);
            }
            INDArray dldzCurrent = a.backprop(zCurrent, dldaCurrent).getFirst();

            //weight gradients:
            Nd4j.gemm(inCurrent, dldzCurrent, wg, true, false, 1.0, 1.0);

            //Recurrent weight gradients:
            Nd4j.gemm(aLast, dldzCurrent, rwg, true, false, 1.0, 1.0);

            //Bias gradients
            bg.addi(dldzCurrent.sum(0));

            //Epsilon out to layer below (i.e., dLdIn)
            Nd4j.gemm(dldzCurrent, w, epsOutCurrent, false, true, 1.0, 0.0);

            dldzNext = dldzCurrent;
        }

        weightNoiseParams.clear();

        Gradient g = new DefaultGradient(gradientsFlattened);
        g.gradientForVariable().put(SimpleRnnParamInitializer.WEIGHT_KEY, wg);
        g.gradientForVariable().put(SimpleRnnParamInitializer.RECURRENT_WEIGHT_KEY, rwg);
        g.gradientForVariable().put(SimpleRnnParamInitializer.BIAS_KEY, bg);

        return new Pair<>(g, epsOut);
    }

    @Override
    public boolean isPretrainLayer() {
        return false;
    }

    @Override
    public INDArray preOutput(boolean training){
        return activate(training);
    }

    @Override
    public INDArray activate(boolean training){
        return activateHelper(input, null, training, false).getFirst();
    }

    private Pair<INDArray,INDArray> activateHelper(INDArray in, INDArray prevStepOut, boolean training, boolean forBackprop){

        int tsLength = in.size(1);

        INDArray w = getParam(SimpleRnnParamInitializer.WEIGHT_KEY);
        INDArray rw = getParam(SimpleRnnParamInitializer.RECURRENT_WEIGHT_KEY);
        INDArray b = getParam(SimpleRnnParamInitializer.BIAS_KEY);


        int nOut = layerConf().getNOut();
        INDArray out = Nd4j.createUninitialized(new int[]{in.size(0), nOut, in.size(2)}, 'f');
        INDArray out2d = out.reshape('f', out.size(0)*out.size(2), out.size(1));

        INDArray outZ = (forBackprop ? Nd4j.createUninitialized(out.shape()) : null);

        if(in.ordering() != 'f' || Shape.strideDescendingCAscendingF(in))
            in = in.dup('f');

        INDArray in2d = in.reshape('f', in.size(0)*in.size(2), in.size(1));

        //Minor performance optimization: do the "add bias" first:
        Nd4j.getExecutioner().exec(new BroadcastCopyOp(out2d, b, out2d, 1));

        //Input mmul across time: [minibatch*tsLength, nIn] * [nIn,nOut] = [minibatch*tsLenth, nOut]
        Nd4j.gemm(in2d, w, out2d, false ,false, 1.0, 1.0 ); //beta=1.0 to keep previous biases...

        IActivation a = layerConf().getActivationFn();

        for( int i=0; i<tsLength; i++ ){
            INDArray currOut = out.get(all(), all(), point(i)); //F order
            if(i > 0 || prevStepOut != null){
                //out = activationFn(in*w + last*rw + bias)
//                currOut.addi(prevStepOut.mmu)
                Nd4j.gemm(prevStepOut, rw, currOut, false, false, 1.0, 1.0);
            }

            a.getActivation(currOut, training);

            if(forBackprop){
                outZ.get(all(), all(), point(i)).assign(currOut);
            }

            prevStepOut = currOut;
        }

        //Apply mask, if present:

        return new Pair<>(out, outZ);
    }



}
