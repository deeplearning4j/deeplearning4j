/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.nn.layers.recurrent;


import java.util.*;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.berkeley.Triple;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.deeplearning4j.nn.params.LSTMParamInitializer;
import org.deeplearning4j.optimize.Solver;
import org.deeplearning4j.util.Dropout;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import static org.nd4j.linalg.indexing.NDArrayIndex.interval;
import static org.nd4j.linalg.ops.transforms.Transforms.exp;
import static org.nd4j.linalg.ops.transforms.Transforms.log;
import static org.nd4j.linalg.ops.transforms.Transforms.pow;
import static org.nd4j.linalg.ops.transforms.Transforms.sigmoid;
import static org.nd4j.linalg.ops.transforms.Transforms.tanh;

/**
 * LSTM recurrent net.
 *
 * Based on karpathy et. al's
 * work on generation of image descriptions.
 *
 * @author Adam Gibson
 */
public class LSTM extends BaseLayer<org.deeplearning4j.nn.conf.layers.LSTM> {
    //recurrent weights (iFogZ = iFog & iFogA = iFogF & memCellActivations = c & outputActivations = hOut)
    private INDArray iFogZ, iFogA, memCellActivations, hIn, outputActivations;
    // update values for drop connect
    private INDArray u, u2;
    //current input // paper has it as image representations
    private INDArray xi;
    //predicted time series // paper has it as word representations
    private INDArray xs;

    public LSTM(NeuralNetConfiguration conf) {
        super(conf);
    }

    public LSTM(NeuralNetConfiguration conf, INDArray input) {
        super(conf, input);
    }


    /**
     * SetInput when img and words exist
     * @param xi the current example
     * @param xs the tim series to predict based on
     * @return
     */
    public void setInput(INDArray xi, INDArray xs) {
        this.xi = xi;
        this.xs = xs;
        setInput(Nd4j.vstack(xi,xs));
    }

    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon) {
        // TODO fix following backprop to just backpropGradient - how to use epsilon?

        INDArray activations = activate(true);

        INDArray inputWeights = getParam(LSTMParamInitializer.INPUT_WEIGHT_KEY);
        INDArray recurrentWeights = getParam(LSTMParamInitializer.RECURRENT_WEIGHT_KEY);

        // Original code with y - output passed in...
        //add column of zeros since not used in forward pass
        INDArray dY = Nd4j.vstack(Nd4j.zeros(activations.columns()), activations);

        //backprop the decoder
        INDArray inputWeightGradients = outputActivations.transpose().mmul(dY); //dWd -- TODO is this epsilon?
        INDArray biasGradients = Nd4j.sum(inputWeightGradients,0); // dbd
        INDArray dHout = dY.mmul(inputWeights.transpose()); //TODO is this nextEpsilon?

        if(conf.isUseDropConnect() & conf.getLayer().getDropOut() > 0)
            dHout.muli(u2);

        //backprop the LSTM
        INDArray dIFogZ = Nd4j.zeros(iFogZ.shape());
        INDArray dIFogA = Nd4j.zeros(iFogA.shape());
        INDArray recurrentWeightGradients = Nd4j.zeros(recurrentWeights.shape()); //dWLSTM
        INDArray dHin = Nd4j.zeros(hIn.shape());

        INDArray dC = Nd4j.zeros(memCellActivations.shape());
        INDArray dX = Nd4j.zeros(input.shape());

        int sequenceLen = outputActivations.rows(); // n
        int hiddenLayerSize = outputActivations.columns(); // d


        for(int t = sequenceLen -1; t > 0; t--) {
            if(conf.getLayer().getActivationFunction().equals("tanh")) {
                INDArray tanhCt = tanh(memCellActivations.slice(t));
                dIFogA.slice(t).put(new INDArrayIndex[]{interval(2 * hiddenLayerSize,3 * hiddenLayerSize)},tanhCt.mul(dHout.slice(t)));
                dC.slice(t).addi(pow(tanhCt,2).rsubi(1).muli(iFogA.slice(t).get(interval(2 * hiddenLayerSize, 3 * hiddenLayerSize)).mul(dHout.slice(t))));
            }
            else {
                dIFogA.slice(t).put(new INDArrayIndex[]{interval(2 * hiddenLayerSize,3 * hiddenLayerSize)},memCellActivations.slice(t).mul(dHout.slice(t)));
                dC.slice(t).addi(iFogA.slice(t).get(interval(2 * hiddenLayerSize,3 * hiddenLayerSize)).mul(dHout.slice(t)));
            }

            if(t > 0) {
                dIFogA.slice(t).put(new INDArrayIndex[]{interval(hiddenLayerSize, 2 * hiddenLayerSize)},memCellActivations.slice(t - 1).mul(dC.slice(t)));
                dC.slice(t - 1).addi(iFogA.slice(t).get(interval(hiddenLayerSize, 2 * hiddenLayerSize)).mul(dC.slice(t)));
            }
            dIFogA.slice(t).put(new INDArrayIndex[]{interval(0, hiddenLayerSize)}, iFogA.slice(t).get(interval(3 * hiddenLayerSize, iFogA.columns())).mul(dC.slice(t)));
            dIFogA.slice(t).put(new INDArrayIndex[]{interval(3 * hiddenLayerSize, dIFogA.columns())},iFogA.slice(t).get(interval(0,hiddenLayerSize)).mul(dC.slice(t)));

            //backprop activation functions
            dIFogZ.slice(t).put(new INDArrayIndex[]{interval(3 * hiddenLayerSize, dIFogZ.columns())},pow(iFogA.slice(t).get(interval(3 * hiddenLayerSize,iFogA.columns())),2).rsubi(1).mul(dIFogA.slice(t).get(interval(3 * hiddenLayerSize,dIFogA.columns()))));
            activations = iFogA.slice(t).get(interval(0,3 * hiddenLayerSize));
            dIFogA.slice(t).put(new INDArrayIndex[]{interval(0, 3 * hiddenLayerSize)}, activations.mul(activations.rsub(1)).mul(dIFogA.slice(t).get(interval(0, 3 * hiddenLayerSize))));

            //backprop matrix multiply
            recurrentWeightGradients.addi(hIn.slice(t).transpose().mmul(dIFogZ.slice(t)));
            dHin.slice(t).assign(dIFogZ.slice(t).mmul(recurrentWeights.transpose()));

            INDArray get = dHin.slice(t).get(interval(1, 1 + hiddenLayerSize));
            dX.slice(t).assign(get);
            if(t > 0)
                dHout.slice(t - 1).addi(dHin.slice(t).get(interval(1 + hiddenLayerSize, dHin.columns())));


            if(conf.isUseDropConnect() & conf.getLayer().getDropOut() > 0)
                dX.muli(u);

        }

        clear(); //TODO is this still needed

        Gradient retGradient = new DefaultGradient();
        retGradient.gradientForVariable().put(LSTMParamInitializer.INPUT_WEIGHT_KEY, inputWeightGradients);
        retGradient.gradientForVariable().put(LSTMParamInitializer.RECURRENT_WEIGHT_KEY, recurrentWeightGradients);
        retGradient.gradientForVariable().put(LSTMParamInitializer.BIAS_KEY, biasGradients);


        return new Pair<>(retGradient, inputWeightGradients);

    }


    @Override
    public INDArray activate(boolean training) {
        INDArray prevOutputActivations, prevMemCellActivations;

        INDArray decoderWeights = getParam(LSTMParamInitializer.INPUT_WEIGHT_KEY);
        INDArray recurrentWeights = getParam(LSTMParamInitializer.RECURRENT_WEIGHT_KEY);
        INDArray decoderBias = getParam(LSTMParamInitializer.BIAS_KEY);


        if(conf.getLayer().getDropOut() > 0) {
            double scale = 1 / (1 - conf.getLayer().getDropOut());
            u = Nd4j.rand(input.shape()).lti(1 - conf.getLayer().getDropOut()).muli(scale);
            input.muli(u);
        }

        int sequenceLen = input.rows(); // n, not miniBatch
        int hiddenLayerSize = decoderWeights.rows(); // hidden layer size
        int recurrentSize = recurrentWeights.size(0);

        hIn = Nd4j.zeros(sequenceLen, recurrentWeights.rows()); //xt, ht-1, bias
        outputActivations = Nd4j.zeros(sequenceLen, hiddenLayerSize);
        //non linearities
        iFogZ = Nd4j.zeros(sequenceLen, hiddenLayerSize * 4);
        iFogA = Nd4j.zeros(iFogZ.shape());
        memCellActivations = Nd4j.zeros(sequenceLen, hiddenLayerSize);


        for(int t = 0; t < sequenceLen ; t++) {
            prevOutputActivations = t == 0 ? Nd4j.zeros(hiddenLayerSize) : outputActivations.getRow(t - 1);
            prevMemCellActivations = t == 0 ? Nd4j.zeros(hiddenLayerSize) : memCellActivations.getRow(t - 1);

            hIn.put(t, 0, 1.0);
            hIn.slice(t).put(new INDArrayIndex[]{interval(1, 1 + hiddenLayerSize)}, input.slice(t));
            hIn.slice(t).put(new INDArrayIndex[]{interval(1 + hiddenLayerSize, hIn.columns())}, prevOutputActivations);

            //compute all gate activations. dots:
            iFogZ.putRow(t, hIn.slice(t).mmul(recurrentWeights));

            //store activations for i, f, o
            iFogA.slice(t).put(new INDArrayIndex[]{interval(0, 3 * hiddenLayerSize)}, sigmoid(iFogZ.slice(t).get(new INDArrayIndex[]{interval(0, 3 * hiddenLayerSize)})));

            // store activations for c
            iFogA.slice(t).put(new INDArrayIndex[]{interval(3 * hiddenLayerSize, iFogA.columns() - 1)},
                    tanh(iFogZ.slice(t).get(interval(3 * hiddenLayerSize, iFogZ.columns() - 1))));

            //i dot product h(WcxXt + WcmMt-1)
            memCellActivations.putRow(t, iFogA.slice(t).get(interval(0, hiddenLayerSize)).mul(iFogA.slice(t).get(interval(3 * hiddenLayerSize, iFogA.columns()))));


            if(t > 0)
                // Ct curr memory cell activations after t 0
                memCellActivations.slice(t).addi(iFogA.slice(t).get(interval(hiddenLayerSize, 2 * hiddenLayerSize)).mul(prevMemCellActivations));

            // mt hidden out or output before activation
            if(conf.getLayer().getActivationFunction().equals("tanh")) {
                outputActivations.slice(t).assign(iFogA.slice(t).get(interval(2 * hiddenLayerSize, 3 * hiddenLayerSize)).mul(tanh(memCellActivations.getRow(t))));
            } else {
                outputActivations.slice(t).assign(iFogA.slice(t).get(interval(2 * hiddenLayerSize, 3 * hiddenLayerSize)).mul(memCellActivations.getRow(t)));
            }
        }

        if(conf.isUseDropConnect() && training) {
            if (conf.getLayer().getDropOut() > 0) {
                u2 = Dropout.applyDropout(outputActivations, conf.getLayer().getDropOut(), u2);
                outputActivations.muli(u2);
            }
        }

        return outputActivations.get(interval(1, outputActivations.rows())).mmul(decoderWeights).addiRowVector(decoderBias);


    }

    /**
     * Prediction with beam search
     * @param xi
     * @param ws
     * @return
     */
    //TODO is this deprecated ?
    public Collection<Pair<List<Integer>,Double>> predict(INDArray xi,INDArray ws) {
        INDArray decoderWeights = getParam(LSTMParamInitializer.INPUT_WEIGHT_KEY);
        int d = decoderWeights.rows();
        Triple<INDArray,INDArray,INDArray> yhc = lstmTick(xi, Nd4j.zeros(d), Nd4j.zeros(d));
        BeamSearch search = new BeamSearch(20,ws,yhc.getSecond(),yhc.getThird());
        return search.search();
    }

    //TODO is this deprecated ?
    @Override
    public  void clear() {
        hIn = null;
        input = null;
        iFogZ = null;
        iFogA = null;
        u = null;
        u2 = null;
        memCellActivations = null;
        outputActivations = null;
    }

    //TODO is this deprecated ?
    private  class BeamSearch {
        private List<Beam> beams = new ArrayList<>();
        private int nSteps = 0;
        private INDArray h,c;
        private INDArray ws;
        private int beamSize = 5;
        public BeamSearch(int nSteps,INDArray ws, INDArray h, INDArray c) {
            this.nSteps = nSteps;
            this.h = h;
            this.c = c;
            this.ws = ws;
            beams.add(new Beam(0.0,new ArrayList<Integer>(),h,c));

        }

        public Collection<Pair<List<Integer>,Double>> search() {
            if(beamSize > 1) {
                while(true) {
                    List<Beam> candidates = new ArrayList<>();
                    for(Beam beam : beams) {
                        //  ixprev = b[1][-1] if b[1] else 0 # start off with the word where this beam left off
                        int ixPrev = beam.getIndices().get(beam.getIndices().size() - 1);
                        if(ixPrev == 0 && !beam.getIndices().isEmpty()) {
                            candidates.add(beam);
                            continue;
                        }

                        Triple<INDArray,INDArray,INDArray> yhc = lstmTick(ws.slice(ixPrev),beam.getHidden(),beam.getC());
                        INDArray y1 = yhc.getFirst().ravel();
                        double maxy1 = y1.max(Integer.MAX_VALUE).getDouble(0);
                        INDArray e1 = exp(y1.subi(maxy1));
                        INDArray p1 = e1.divi(Nd4j.sum(e1,Integer.MAX_VALUE));
                        y1 = log(p1.addi(Nd4j.EPS_THRESHOLD));
                        //indices/sorted arrays
                        INDArray[] topIndices = Nd4j.sortWithIndices(y1,0,false);
                        for(int i = 0; i < beamSize; i++) {
                            int idx = topIndices[0].getInt(i);
                            List<Integer> beamCopy = new ArrayList<>(beam.getIndices());
                            beamCopy.add(idx);
                            candidates.add(new Beam(beam.getLogProba() + y1.getDouble(idx),beamCopy,yhc.getSecond(),yhc.getThird()));
                        }


                    }

                    //sort the beams
                    //truncate beams to be of beam size also setting beams = candidates
                    nSteps++;
                    if(nSteps >= 20)
                        break;

                }

                List<Pair<List<Integer>,Double>> ret = new ArrayList<>();
                for(Beam b : beams) {
                    ret.add(new Pair<>(b.getIndices(),b.getLogProba()));
                }

                return ret;


            }

            else {
                int ixPrev = 0;
                double predictedLogProba = 0.0;
                List<Integer> predix = new ArrayList<>();
                while(true) {
                    Triple<INDArray,INDArray,INDArray> yhc = lstmTick(ws.slice(ixPrev),h,c);
                    Pair<Integer,Double> yMax = yMax(yhc.getFirst());
                    predix.add(yMax.getFirst());
                    predictedLogProba += yMax.getSecond();

                    nSteps++;
                    if(ixPrev == 0 || nSteps >= 20)
                        break;
                }

                return Collections.singletonList(new Pair<>(predix,predictedLogProba));

            }

        }
    }

    //TODO is this deprecated ?
    private Pair<Integer,Double> yMax(INDArray y) {
        INDArray y1 = y.linearView();
        double max = y.max(Integer.MAX_VALUE).getDouble(0);
        INDArray e1 = exp(y1.rsub(max));
        INDArray p1 = e1.divi(e1.sum(Integer.MAX_VALUE));
        y1 = log(p1.addi(Nd4j.EPS_THRESHOLD));
        INDArray[] sorted = Nd4j.sortWithIndices(y1,0,true);
        int ix = sorted[0].getInt(0);
        return new Pair<>(ix,sorted[1].getDouble(ix));
    }

    //TODO is this deprecated ?
    private static class Beam {
        private double logProba = 0.0;
        private List<Integer> indices;
        //hidden and cell states
        private INDArray hidden,c;

        public Beam(double logProba, List<Integer> indices, INDArray hidden, INDArray c) {
            this.logProba = logProba;
            this.indices = indices;
            this.hidden = hidden;
            this.c = c;
        }

        public double getLogProba() {
            return logProba;
        }

        public void setLogProba(double logProba) {
            this.logProba = logProba;
        }

        public List<Integer> getIndices() {
            return indices;
        }

        public void setIndices(List<Integer> indices) {
            this.indices = indices;
        }

        public INDArray getHidden() {
            return hidden;
        }

        public void setHidden(INDArray hidden) {
            this.hidden = hidden;
        }

        public INDArray getC() {
            return c;
        }

        public void setC(INDArray c) {
            this.c = c;
        }
    }

    private Triple<INDArray,INDArray,INDArray> lstmTick(INDArray x,INDArray hPrev,INDArray cPrev) {
        INDArray decoderWeights = getParam(LSTMParamInitializer.INPUT_WEIGHT_KEY);
        INDArray recurrentWeights = getParam(LSTMParamInitializer.RECURRENT_WEIGHT_KEY);
        INDArray decoderBias = getParam(LSTMParamInitializer.BIAS_KEY);

        int t = 0;
        int d = decoderWeights.rows();
        INDArray hIn = Nd4j.zeros(1,recurrentWeights.rows());
        hIn.putRow(0,Nd4j.ones(hIn.columns()));
        hIn.slice(t).put(new INDArrayIndex[]{interval(1,1 + d)},x);
        hIn.slice(t).put(new INDArrayIndex[]{interval(1 + d,hIn.columns())},hPrev);


        INDArray iFog = Nd4j.zeros(1, d * 4);
        INDArray iFogf = Nd4j.zeros(iFog.shape());
        INDArray c = Nd4j.zeros(d);
        iFog.putScalar(t,hIn.slice(t).mmul(recurrentWeights).getDouble(0));
        INDArrayIndex[] indices = new INDArrayIndex[]{interval(0,3 * d)};
        iFogf.slice(t).put(indices,sigmoid(iFogA.slice(t).get(indices)));
        INDArrayIndex[] after = new INDArrayIndex[]{interval(3 * d,iFogf.columns())};
        iFogf.slice(t).put(after,tanh(iFogf.slice(t).get(after)));
        c.slice(t).assign(iFogf.slice(t).get(interval(0,d)).mul(iFogf.slice(t).get(interval(3 * d,iFogf.columns()))).addi(iFogf.slice(t).get(interval(d, 2 * d))).muli(cPrev));

        if(conf.getLayer().getActivationFunction().equals("tanh"))
            outputActivations.slice(t).assign(iFogf.slice(t).get(interval(2 * d,3 * d)).mul(tanh(c.slice(t))));
        else
            outputActivations.slice(t).assign(iFogf.slice(t).get(interval(2 * d,3 * d)).mul(c.slice(t)));
        INDArray y = outputActivations.mmul(decoderWeights).addiRowVector(decoderBias);
        return new Triple<>(y,outputActivations,c);


    }

    @Override
    public double calcL2() {
    	if(!conf.isUseRegularization() || conf.getL2() <= 0.0 ) return 0.0;
    	double l2 = Transforms.pow(getParam(LSTMParamInitializer.RECURRENT_WEIGHT_KEY), 2).sum(Integer.MAX_VALUE).getDouble(0)
    			+ Transforms.pow(getParam(LSTMParamInitializer.INPUT_WEIGHT_KEY), 2).sum(Integer.MAX_VALUE).getDouble(0);
    	return 0.5 * conf.getL2() * l2;
    }

    @Override
    public double calcL1() {
    	if(!conf.isUseRegularization() || conf.getL1() <= 0.0 ) return 0.0;
        double l1 = Transforms.abs(getParam(LSTMParamInitializer.RECURRENT_WEIGHT_KEY)).sum(Integer.MAX_VALUE).getDouble(0)
        		+ Transforms.abs(getParam(LSTMParamInitializer.INPUT_WEIGHT_KEY)).sum(Integer.MAX_VALUE).getDouble(0);
        return conf.getL1() * l1;
    }

    @Override
    public Type type(){
        return Type.RECURRENT;
    }

    @Override
    public Layer transpose(){
        throw new UnsupportedOperationException("Not yet implemented");
    }

    //TODO verfiy this is still needed
    @Override
    public void fit(INDArray data) {
        xi = data.slice(0);
        INDArrayIndex[] everythingElse = {
                NDArrayIndex.interval(1,data.rows()),NDArrayIndex.interval(0,data.columns())
        };
        xs = data.get(everythingElse);
        Solver solver = new Solver.Builder()
                .configure(conf).model(this).listeners(getListeners())
                .build();
        solver.optimize();
    }

    //TODO verfiy this is correct
    @Override
    public int batchSize() {
        return xi.rows();
    }

}
