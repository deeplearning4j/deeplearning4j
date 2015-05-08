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


import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.berkeley.Triple;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.deeplearning4j.nn.params.LSTMParamInitializer;
import org.deeplearning4j.optimize.Solver;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.lossfunctions.LossFunctions;

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
public class LSTM extends BaseLayer {
    //recurrent weights
    private INDArray iFog,iFogF,c,x,hIn,hOut,u,u2;
    //current input
    private INDArray xi;
    //predicted time series
    private INDArray xs;

    public LSTM(NeuralNetConfiguration conf) {
        super(conf);
    }



    /**
     * Forward propagation
     * @param xi the current example
     * @param xs the tim series to predict based on
     * @return
     */
    public INDArray forward(INDArray xi,INDArray xs) {
        this.xs = xs;
        this.xi = xi;
        x = Nd4j.vstack(xi,xs);
        return activate(x);
    }


    /**
     * Back propagation in the given input
     * @param y
     * @return {@link org.deeplearning4j.nn.gradient.Gradient}
     */
    public Gradient backward(INDArray y) {
        INDArray decoderWeights = getParam(LSTMParamInitializer.DECODER_WEIGHTS);
        INDArray recurrentWeights = getParam(LSTMParamInitializer.RECURRENT_WEIGHTS);

        INDArray dY = Nd4j.vstack(Nd4j.zeros(y.columns()),y);
        INDArray dWd = hOut.transpose().mmul(dY);
        INDArray dBd = Nd4j.sum(dWd,0);
        INDArray dHout = dY.mmul(decoderWeights.transpose());
        if(conf.getDropOut() > 0) {
            dHout.muli(u2);
        }



        INDArray dIFog = Nd4j.zeros(iFog.shape());
        INDArray dIFogF = Nd4j.zeros(iFogF.shape());
        INDArray dRecurrentWeights = Nd4j.zeros(recurrentWeights.shape());
        INDArray dHin = Nd4j.zeros(hIn.shape());

        INDArray dC = Nd4j.zeros(c.shape());
        INDArray dx = Nd4j.zeros(x.shape());
        int n = hOut.rows();
        int d = hOut.columns();


        for(int t = n -1; t > 0; t--) {
            if(conf.getActivationFunction().equals("tanh")) {
                INDArray tanhCt = tanh(c.slice(t));
                dIFogF.slice(t).put(new NDArrayIndex[]{interval(2 * d,3 * d)},tanhCt.mul(dHout.slice(t)));
                dC.slice(t).addi(pow(tanhCt,2).rsubi(1).muli(iFogF.slice(t).get(interval(2 * d, 3 * d)).mul(dHout.slice(t))));
            }
            else {
                dIFogF.slice(t).put(new NDArrayIndex[]{interval(2 * d,3 * d)},c.slice(t).mul(dHout.slice(t)));
                dC.slice(t).addi(iFogF.slice(t).get(interval(2 * d,3 * d)).mul(dHout.slice(t)));
            }

            if(t > 0) {
                dIFogF.slice(t).put(new NDArrayIndex[]{interval(d, 2 * d)},c.slice(t - 1).mul(dC.slice(t)));
                dC.slice(t - 1).addi(iFogF.slice(t).get(interval(d,2 * d)).mul(dC.slice(t)));
            }

            dIFogF.slice(t).put(new NDArrayIndex[]{interval(0, d)}, iFogF.slice(t).get(interval(3 * d, iFogF.columns())).mul(dC.slice(t)));
            dIFogF.slice(t).put(new NDArrayIndex[]{interval(3 * d, dIFogF.columns())},iFogF.slice(t).get(interval(0,d)).mul(dC.slice(t)));

            dIFog.slice(t).put(new NDArrayIndex[]{interval(3 * d,dIFog.columns())},pow(iFogF.slice(t).get(interval(3 * d,iFogF.columns())),2).rsubi(1).mul(dIFogF.slice(t).get(interval(3 * d,dIFogF.columns()))));
            y = iFogF.slice(t).get(interval(0,3 * d));
            dIFogF.slice(t).put(new NDArrayIndex[]{interval(0, 3 * d)}, y.mul(y.rsub(1)).mul(dIFogF.slice(t).get(interval(0, 3 * d))));

            dRecurrentWeights.addi(hIn.slice(t).transpose().mmul(dIFog.slice(t)));
            dHin.slice(t).assign(dIFog.slice(t).mmul(recurrentWeights.transpose()));


            INDArray get = dHin.slice(t).get(interval(1, 1 + d));

            dx.slice(t).assign(get);
            if(t > 0)
                dHout.slice(t - 1).addi(dHin.slice(t).get(interval(1 + d, dHin.columns())));



            if(conf.getDropOut() > 0)
                dx.muli(u);

        }


        clear();

        Gradient gradient = new DefaultGradient();
        gradient.gradientForVariable().put(LSTMParamInitializer.DECODER_BIAS,dBd);
        gradient.gradientForVariable().put(LSTMParamInitializer.DECODER_WEIGHTS,dWd);
        gradient.gradientForVariable().put(LSTMParamInitializer.RECURRENT_WEIGHTS,dRecurrentWeights);
        return gradient;

    }


    @Override
    public INDArray activate(INDArray input) {
        INDArray decoderWeights = getParam(LSTMParamInitializer.DECODER_WEIGHTS);
        INDArray recurrentWeights = getParam(LSTMParamInitializer.RECURRENT_WEIGHTS);
        INDArray decoderBias = getParam(LSTMParamInitializer.DECODER_BIAS);


        if(conf.getDropOut() > 0) {
            double scale = 1 / (1 - conf.getDropOut());
            u = Nd4j.rand(x.shape()).lti(1 - conf.getDropOut()).muli(scale);
            x.muli(u);
        }

        int n = x.rows();
        int d = decoderWeights.rows();
        //xt, ht-1, bias
        hIn = Nd4j.zeros(n,recurrentWeights.rows());
        hOut = Nd4j.zeros(n,d);
        //non linearities
        iFog = Nd4j.zeros(n,d * 4);
        iFogF = Nd4j.zeros(iFog.shape());
        c = Nd4j.zeros(n,d);

        INDArray prev;

        for(int t = 0; t < n ; t++) {
            prev = t == 0 ? Nd4j.zeros(d) : hOut.getRow(t - 1);
            hIn.put(t, 0, 1.0);
            hIn.slice(t).put(new NDArrayIndex[]{interval(1,1 + d)},x.slice(t));
            hIn.slice(t).put(new NDArrayIndex[]{interval(1 + d,hIn.columns())},prev);

            //compute all gate activations. dots:
            iFog.putRow(t,hIn.slice(t).mmul(recurrentWeights));

            //non linearity
            iFogF.slice(t).put(new NDArrayIndex[]{interval(0,3 * d)}, sigmoid(iFog.slice(t).get(interval(0, 3 * d))));
            iFogF.slice(t).put(new NDArrayIndex[]{interval(3 * d,iFogF.columns() - 1)}, tanh(iFog.slice(t).get(interval(3 * d, iFog.columns() - 1))));

            //cell activations
            INDArray cPut = iFogF.slice(t).get(interval(0, d)).mul(iFogF.slice(t).get(interval(3 * d, iFogF.columns())));
            c.putRow(t,cPut);


            if(t > 0)
                c.slice(t).addi(iFogF.slice(t).get(interval(d,2 * d)).mul(c.getRow(t - 1)));


            if(conf.getActivationFunction().equals("tanh"))
                hOut.slice(t).assign(iFogF.slice(t).get(interval(2 * d,3 * d)).mul(tanh(c.getRow(t))));

            else
                hOut.slice(t).assign(iFogF.slice(t).get(interval(2 * d,3 * d)).mul(c.getRow(t)));




        }

        if(conf.getDropOut() > 0) {
            double scale = 1 / (1 - conf.getDropOut());
            u2 = Nd4j.rand(hOut.shape()).lti(1 - conf.getDropOut()).muli(scale);
            hOut.muli(u2);
        }


        INDArray y = hOut.get(interval(1,hOut.rows())).mmul(decoderWeights).addiRowVector(decoderBias);
        return y;


    }

    /**
     * Prediction with beam search
     * @param xi
     * @param ws
     * @return
     */
    public Collection<Pair<List<Integer>,Double>> predict(INDArray xi,INDArray ws) {
        INDArray decoderWeights = getParam(LSTMParamInitializer.DECODER_WEIGHTS);
        int d = decoderWeights.rows();
        Triple<INDArray,INDArray,INDArray> yhc = lstmTick(xi,Nd4j.zeros(d),Nd4j.zeros(d));
        BeamSearch search = new BeamSearch(20,ws,yhc.getSecond(),yhc.getThird());
        return search.search();
    }


    @Override
    public  void clear() {
        u = null;
        hIn = null;
        hOut = null;
        iFog = null;
        iFogF = null;
        c = null;
        x = null;
        u2 = null;
    }


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
        INDArray decoderWeights = getParam(LSTMParamInitializer.DECODER_WEIGHTS);
        INDArray recurrentWeights = getParam(LSTMParamInitializer.RECURRENT_WEIGHTS);
        INDArray decoderBias = getParam(LSTMParamInitializer.DECODER_BIAS);

        int t = 0;
        int d = decoderWeights.rows();
        INDArray hIn = Nd4j.zeros(1,recurrentWeights.rows());
        hIn.putRow(0,Nd4j.ones(hIn.columns()));
        hIn.slice(t).put(new NDArrayIndex[]{interval(1,1 + d)},x);
        hIn.slice(t).put(new NDArrayIndex[]{interval(1 + d,hIn.columns())},hPrev);


        INDArray iFog = Nd4j.zeros(1, d * 4);
        INDArray iFogf = Nd4j.zeros(iFog.shape());
        INDArray c = Nd4j.zeros(d);
        iFog.putScalar(t,hIn.slice(t).mmul(recurrentWeights).getDouble(0));
        NDArrayIndex[] indices = new NDArrayIndex[]{interval(0,3 * d)};
        iFogf.slice(t).put(indices,sigmoid(iFogF.slice(t).get(indices)));
        NDArrayIndex[] after = new NDArrayIndex[]{interval(3 * d,iFogf.columns())};
        iFogf.slice(t).put(after,tanh(iFogf.slice(t).get(after)));
        c.slice(t).assign(iFogf.slice(t).get(interval(0,d)).mul(iFogf.slice(t).get(interval(3 * d,iFogf.columns()))).addi(iFogf.slice(t).get(interval(d, 2 * d))).muli(cPrev));

        if(conf.getActivationFunction().equals("tanh"))
            hOut.slice(t).assign(iFogf.slice(t).get(interval(2 * d,3 * d)).mul(tanh(c.slice(t))));
        else
            hOut.slice(t).assign(iFogf.slice(t).get(interval(2 * d,3 * d)).mul(c.slice(t)));
        INDArray y = hOut.mmul(decoderWeights).addiRowVector(decoderBias);
        return new Triple<>(y,hOut,c);


    }


    @Override
    public void fit() {
        Solver solver = new Solver.Builder()
                .model(this).configure(conf()).listeners(getIterationListeners())
                .build();
        solver.optimize();
    }



    @Override
    public void update(Gradient gradient) {
        setParams(params().addi(gradient.gradient()));
    }

    @Override
    public double score() {
        INDArray forward =  Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("softmax", forward(xi,xs)).derivative(), 1);
        return LossFunctions.score(xs,conf.getLossFunction(),forward,conf.getL2(),conf.isUseRegularization());
    }

    @Override
    public INDArray transform(INDArray data) {
        return  Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("softmax", forward(xi,xs)).derivative(), 1);
    }



    @Override
    public void setParams(INDArray params) {
        int count = 0;
        INDArray decoderWeights = getParam(LSTMParamInitializer.DECODER_WEIGHTS);
        INDArray recurrentWeights = getParam(LSTMParamInitializer.RECURRENT_WEIGHTS);
        INDArray decoderBias = getParam(LSTMParamInitializer.DECODER_BIAS);

        INDArray recurrentWeightsLinear = recurrentWeights.linearView();
        INDArray decoderWeightsLinear = decoderWeights.linearView();
        INDArray decoderBiasLinear = decoderBias.linearView();
        int recurrentPlusDecoder = recurrentWeightsLinear.length() + decoderWeightsLinear.length();
        boolean pastRecurrentWeights = false;
        for(int i = 0; i < params.length(); i++) {
            //reset once for normal recurrent weights
            if(count == recurrentWeightsLinear.length()) {
                count = 0;
                pastRecurrentWeights = true;
            }
            //reset again for decoder weights, no need to do this as this sets up the bias count properly
            else if(count == decoderWeightsLinear.length() && pastRecurrentWeights)
                count = 0;

            if(i < recurrentWeights.length())
                recurrentWeights.linearView().putScalar(count++,params.getDouble(i));

            else if(i < recurrentPlusDecoder)
                decoderWeightsLinear.putScalar(count++,params.getDouble(i));
            else
                decoderBiasLinear.putScalar(count++,params.getDouble(i));

        }
    }

    @Override
    public void fit(INDArray data) {
        xi = data.slice(0);
        NDArrayIndex[] everythingElse = {
                NDArrayIndex.interval(1,data.rows()),NDArrayIndex.interval(0,data.columns())
        };
        xs = data.get(everythingElse);
        Solver solver = new Solver.Builder()
                .configure(conf).model(this).listeners(getIterationListeners())
                .build();
        solver.optimize();
    }

    @Override
    public void iterate(INDArray input) {

    }

    @Override
    public Gradient gradient() {
        INDArray forward = forward(xi,xs);
        INDArray probas = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("softmax",forward).derivative(),1);
        return backward(probas);
    }

    @Override
    public Pair<Gradient, Double> gradientAndScore() {
        return new Pair<>(gradient(),score());
    }

    @Override
    public int batchSize() {
        return xi.rows();
    }

}
