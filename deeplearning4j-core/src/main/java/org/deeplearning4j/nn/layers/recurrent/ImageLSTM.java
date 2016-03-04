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


import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.berkeley.Triple;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.deeplearning4j.nn.params.ImageLSTMParamInitializer;
import org.deeplearning4j.optimize.Solver;
import org.deeplearning4j.util.Dropout;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;

import static org.nd4j.linalg.indexing.NDArrayIndex.interval;
import static org.nd4j.linalg.ops.transforms.Transforms.*;

/**
 * LSTM image recurrent net.
 *
 * Based on karpathy et. al's
 * work on generation of image descriptions.
 *
 * @author Adam Gibson
 */
public class ImageLSTM extends BaseLayer<org.deeplearning4j.nn.conf.layers.ImageLSTM> {
    //recurrent weights (iFogZ = iFog & iFogA = iFogF & memCellActivations = c & outputActivations = hOut)
    private INDArray iFogZ, iFogA, memCellActivations, hIn, hOut, outputActivations;
    // update values for drop connect
    private INDArray u, u2;
    //current input // paper has it as image representations
    private INDArray xi;
    //predicted time series // paper has it as word representations
    private INDArray xs;

    public ImageLSTM(NeuralNetConfiguration conf) {
        super(conf);
        throw new UnsupportedOperationException("Layer disabled: Version in development and will be provided in a later release.");

    }

    public ImageLSTM(NeuralNetConfiguration conf, INDArray input) {
        super(conf, input);
        throw new UnsupportedOperationException("Layer disabled: Version in development and will be provided in a later release.");
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

    public Pair<Gradient, INDArray> backpropGradient(Gradient gradient, INDArray esilon) {
        // TODO based on associated math this needs to have the previous gradient passed in
        // or needs a separate calculation of decode gradients
        INDArray tanhCt, activations;

        INDArray inputWeights = getParam(ImageLSTMParamInitializer.INPUT_WEIGHT_KEY);
        INDArray recurrentWeights = getParam(ImageLSTMParamInitializer.RECURRENT_WEIGHT_KEY);
        INDArray dHin = Nd4j.zeros(hIn.shape()); // TODO check shape...
        INDArray dX = Nd4j.zeros(input.shape());

         INDArray delta = gradient.getGradientFor(ImageLSTMParamInitializer.BIAS_KEY);
        // TODO pull out delta from input gradient
        INDArray inputWeightGradients = hOut.transpose().mul(delta);
        INDArray biasGradients = Nd4j.sum(delta, 0); // dbd
        //TODO confirm order of multiply - this is epsilon
        INDArray dHout = inputWeights.mul(delta);

        //TODO most layers calc epsilon with previous layers weights but this calc usese current which impacts values and shape
        // add column of zeros since not used in forward pass
        dHout = Nd4j.vstack(Nd4j.zeros(dHout.columns()), dHout);

        if(conf.isUseDropConnect() && conf.getLayer().getDropOut() > 0)
            dHout.muli(u2);

        //backprop the LSTM
        INDArray dIFogZ = Nd4j.zeros(iFogZ.shape());
        INDArray dIFogA = Nd4j.zeros(iFogA.shape());
        INDArray recurrentWeightGradients = Nd4j.zeros(recurrentWeights.shape()); //dWLSTM
        INDArray dC = Nd4j.zeros(memCellActivations.shape());

        int sequenceLen = hOut.rows(); // n
        int hiddenLayerSize = hOut.columns(); // d

        for(int t = sequenceLen -1; t > 0; t--) {
            if(conf.getLayer().getActivationFunction().equals("tanh")) {
                tanhCt = tanh(memCellActivations.slice(t));
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

            //backprop activation functions - this is the derivate of activation? - o first and then i, f, g? below so tanh first and sigmoid after
            dIFogZ.slice(t).put(new INDArrayIndex[]{interval(3 * hiddenLayerSize, dIFogZ.columns())},
                    pow(iFogA.slice(t).get(interval(3 * hiddenLayerSize, iFogA.columns())),2)
                            .rsubi(1).mul(dIFogA.slice(t).get(interval(3 * hiddenLayerSize, dIFogA.columns()))));
            activations = iFogA.slice(t).get(interval(0,3 * hiddenLayerSize));
            dIFogZ.slice(t).put(new INDArrayIndex[]{interval(0, 3 * hiddenLayerSize)},
                    activations.mul(activations.rsub(1)).mul(dIFogA.slice(t).get(interval(0, 3 * hiddenLayerSize))));

            //backprop matrix multiply
            recurrentWeightGradients.addi(hIn.slice(t).transpose().mmul(dIFogZ.slice(t)));
            //TODO verify this equation - its not clear this its used for dHin which becomes dX
//            delta.slice(t).assign(dIFogZ.slice(t).mmul(recurrentWeights.transpose())); //dWd //TODO confirm this is the right approach

            dHin.slice(t).assign(dIFogZ.slice(t).mmul(recurrentWeights.transpose()));
            INDArray get = dHin.slice(t).get(interval(1, 1 + hiddenLayerSize));
            dX.slice(t).assign(get);
            if(t > 0) {
                dHout.slice(t - 1).addi(dHin.slice(t).get(interval(1 + hiddenLayerSize, dHin.columns())));
            }
            if(conf.isUseDropConnect() & conf.getLayer().getDropOut() > 0)
                dX.muli(u);
        }

        //TODO is this still needed?
        clear();
        //backprop the decoder
        Gradient retGradient = new DefaultGradient();
        retGradient.gradientForVariable().put(ImageLSTMParamInitializer.INPUT_WEIGHT_KEY, inputWeightGradients);
        retGradient.gradientForVariable().put(ImageLSTMParamInitializer.RECURRENT_WEIGHT_KEY, recurrentWeightGradients);
        retGradient.gradientForVariable().put(ImageLSTMParamInitializer.BIAS_KEY, biasGradients);


        return new Pair<>(retGradient, dHout);

    }


    @Override
    public INDArray activate(boolean training) {
        INDArray prevOutputActivations, prevMemCellActivations;

        INDArray decoderWeights = getParam(ImageLSTMParamInitializer.INPUT_WEIGHT_KEY);
        INDArray recurrentWeights = getParam(ImageLSTMParamInitializer.RECURRENT_WEIGHT_KEY);
        INDArray decoderBias = getParam(ImageLSTMParamInitializer.BIAS_KEY);


        if(conf.getLayer().getDropOut() > 0) {
            double scale = 1 / (1 - conf.getLayer().getDropOut());
            u = Nd4j.rand(input.shape()).lti(1 - conf.getLayer().getDropOut()).muli(scale);
            input.muli(u);
        }

        int sequenceLen = input.size(0); // n, not miniBatch
        int hiddenLayerSize = decoderWeights.size(0); // hidden layer size
        int recurrentSize = recurrentWeights.size(0);

        hIn = Nd4j.zeros(sequenceLen, recurrentSize); //xt, ht-1, bias
        hOut = Nd4j.zeros(sequenceLen, hiddenLayerSize);
        //non linearities
        iFogZ = Nd4j.zeros(sequenceLen, hiddenLayerSize * 4);
        iFogA = Nd4j.zeros(iFogZ.shape());
        memCellActivations = Nd4j.zeros(sequenceLen, hiddenLayerSize);


        for(int t = 0; t < sequenceLen ; t++) {
            prevOutputActivations = t == 0 ? Nd4j.zeros(hiddenLayerSize) : hOut.slice(t - 1);
            prevMemCellActivations = t == 0 ? Nd4j.zeros(hiddenLayerSize) : memCellActivations.slice(t - 1);

            hIn.slice(t).put(t, 0, 1);
            hIn.slice(t).put(new INDArrayIndex[] {interval(1, 1 + hiddenLayerSize), interval(t,t+1)}, input.slice(t));
            hIn.slice(t).put(new INDArrayIndex[] {interval(1 + hiddenLayerSize, hIn.columns()), interval(0, 1)}, prevOutputActivations);

//            compute all gate activations. dots:
            iFogZ.slice(t).put(new INDArrayIndex[]{interval(0, hiddenLayerSize * 4), interval(0, 1)}, hIn.slice(t).mmul(recurrentWeights));

//            //store activations for i, f, o
            iFogA.slice(t).put(new INDArrayIndex[]{interval(0, 3 * hiddenLayerSize)},
                    sigmoid(iFogZ.slice(t).get(interval(0, 3 * hiddenLayerSize))));

//            // store activations for c
            iFogA.slice(t).put(new INDArrayIndex[]{interval(3 * hiddenLayerSize, iFogA.columns() - 1)},
                    tanh(iFogZ.slice(t).get(interval(3 * hiddenLayerSize, iFogZ.columns() - 1))));

//            //i dot product h(WcxXt + WcmMt-1)
            memCellActivations.slice(t).put(new INDArrayIndex[]{interval(0, hiddenLayerSize)},
                    iFogA.slice(t).get(interval(0, hiddenLayerSize)).mul(iFogA.slice(t).get(interval(3 * hiddenLayerSize, iFogA.columns()))));

            if(t > 0)
//                // Ct curr memory cell activations after t 0
                memCellActivations.slice(t).addi(iFogA.slice(t).get(
                        interval(hiddenLayerSize, 2 * hiddenLayerSize)).mul(prevMemCellActivations));

//            // mt hidden out or output before activation
            if(conf.getLayer().getActivationFunction().equals("tanh")) {
                hOut.slice(t).assign(
                        iFogA.slice(t).get(interval(2 * hiddenLayerSize, 3 * hiddenLayerSize)).mul(tanh(memCellActivations.slice(t))));
            } else {
                hOut.slice(t).assign(
                        iFogA.slice(t).get(interval(2 * hiddenLayerSize, 3 * hiddenLayerSize)).mul(memCellActivations.slice(t)));
            }
        }

        if(conf.isUseDropConnect() && training) {
            if (conf.getLayer().getDropOut() > 0) {
                u2 = Dropout.applyDropout(hOut, conf.getLayer().getDropOut(), u2);
                hOut.muli(u2);
            }
        }

        outputActivations = hOut.get(interval(1, hOut.rows())).mmul(decoderWeights).addiRowVector(decoderBias);
        return outputActivations;


    }

    //TODO review and update
    public Collection<Pair<List<Integer>,Double>> predict(INDArray xi,INDArray ws) {
        INDArray decoderWeights = getParam(ImageLSTMParamInitializer.INPUT_WEIGHT_KEY);
        int d = decoderWeights.rows();
        Triple<INDArray,INDArray,INDArray> yhc = lstmTick(xi, Nd4j.zeros(d), Nd4j.zeros(d));
        BeamSearch search = new BeamSearch(20,ws,yhc.getSecond(),yhc.getThird());
        return search.search();
    }

    //TODO review and update
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

    //TODO review and update
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

    //TODO review and update
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

    //TODO review and update
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
        INDArray decoderWeights = getParam(ImageLSTMParamInitializer.INPUT_WEIGHT_KEY);
        INDArray recurrentWeights = getParam(ImageLSTMParamInitializer.RECURRENT_WEIGHT_KEY);
        INDArray decoderBias = getParam(ImageLSTMParamInitializer.BIAS_KEY);

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
    	if(!conf.isUseRegularization() || conf.getLayer().getL2() <= 0.0 ) return 0.0;
        double l2Norm = getParam(ImageLSTMParamInitializer.RECURRENT_WEIGHT_KEY).norm2Number().doubleValue();
        double sumSquaredWeights = l2Norm*l2Norm;

        l2Norm = getParam(ImageLSTMParamInitializer.INPUT_WEIGHT_KEY).norm2Number().doubleValue();
        sumSquaredWeights += l2Norm*l2Norm;

        return 0.5 * conf.getLayer().getL2() * sumSquaredWeights;
    }

    @Override
    public double calcL1() {
    	if(!conf.isUseRegularization() || conf.getLayer().getL1() <= 0.0 ) return 0.0;
        double l1 = getParam(ImageLSTMParamInitializer.RECURRENT_WEIGHT_KEY).norm1Number().doubleValue()
        		+ getParam(ImageLSTMParamInitializer.INPUT_WEIGHT_KEY).norm1Number().doubleValue();
        return conf.getLayer().getL1() * l1;
    }

    @Override
    public Type type(){
        return Type.RECURRENT;
    }

    @Override
    public Layer transpose(){
        throw new UnsupportedOperationException("Not yet implemented");
    }

    //TODO review and update
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

    //TODO review and update - determine how to work with batch size
    @Override
    public int batchSize() {
        return xi.rows();
    }

}
