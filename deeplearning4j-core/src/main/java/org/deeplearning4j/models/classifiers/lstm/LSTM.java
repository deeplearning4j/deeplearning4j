package org.deeplearning4j.models.classifiers.lstm;


import static org.nd4j.linalg.indexing.NDArrayIndex.interval;
import static org.nd4j.linalg.ops.transforms.Transforms.*;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.berkeley.Triple;
import org.deeplearning4j.nn.WeightInitUtil;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.optimize.Solver;
import org.nd4j.linalg.api.activation.Activations;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.AdaGrad;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;

/**
 * LSTM recurrent net.
 *
 * Based on karpathy et. al's work on generation of image descriptions.
 *
 * @author Adam Gibson
 */
public class LSTM implements Serializable,Model {
    //recurrent weights
    private INDArray recurrentWeights;

    private INDArray decoderWeights;
    private INDArray decoderBias;
    private NeuralNetConfiguration conf;
    private INDArray iFog,iFogF,c,x,hIn,hOut,u,u2;
    private INDArray xi;
    private INDArray xs;


    private void init() {
        //Includex x_t, h_t - 1 and the bias unit
        //output is the 3 gates and the cell signal
        recurrentWeights = WeightInitUtil.initWeights(conf.getnIn() + conf.getnOut() + 1,conf.getnOut() * 4,conf.getWeightInit(),conf.getActivationFunction(),conf.getDist());
        decoderWeights = WeightInitUtil.initWeights(conf.getnOut(),conf.getRecurrentOutput(),conf.getWeightInit(),conf.getActivationFunction(),conf.getDist());
        decoderBias = Nd4j.zeros(conf.getRecurrentOutput());


    }


    public INDArray forward(INDArray xi,INDArray xs) {
        x = Nd4j.vstack(xi,xs);
        this.xs = xs;
        this.xi = xi;
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
            //prev = np.zeros(d) if t == 0 else Hout[t-1]
            prev = t == 0 ? Nd4j.zeros(d) : hOut.getRow(t - 1);
            hIn.put(t, 0, 1.0);
            hIn.slice(t).put(new NDArrayIndex[]{interval(1,1 + d)},hOut.getRow(t - 1));
            hIn.slice(t).put(new NDArrayIndex[]{interval(1 + d,hIn.columns())},prev);

            //compute all gate activations. dots:
            iFog.putRow(t,hIn.slice(t).mmul(recurrentWeights));


            iFogF.slice(t).put(new NDArrayIndex[]{interval(0,3 * d)}, sigmoid(iFog.slice(t).get(interval(0, 3 * d))));
            iFogF.slice(t).put(new NDArrayIndex[]{interval(3 * d,iFogF.columns())}, tanh(iFog.slice(t).get(interval(3 * d, iFog.columns()))));
            c.slice(t).put(new NDArrayIndex[]{interval(3 * d, iFogF.columns())},iFogF.slice(t).get(interval(0, d)).mul(iFogF.slice(t).get(interval(3 * d,iFogF.columns()))));
            if(t > 0)
                c.slice(t).addi(iFogF.slice(t).get(interval(d,2 * d)).mul(c.getRow(t - 1)));


            if(conf.getActivationFunction().type().equals("tanh"))
                hOut.slice(t).assign(iFogF.slice(t).get(interval(2 * d,3 * d)).muli(tanh(c.getRow(t))));

            else
                hOut.slice(t).assign(iFogF.slice(t).get(interval(2 * d,3 * d)).muli(c.getRow(t)));




        }

        if(conf.getDropOut() > 0) {
            double scale = 1 / (1 - conf.getDropOut());
            u2 = Nd4j.rand(hOut.shape()).lti(1 - conf.getDropOut()).muli(scale);
            hOut.muli(u2);
        }


        INDArray y = hOut.getRows(interval(1,hOut.rows()).indices()).mmul(decoderWeights).addiRowVector(decoderBias);
        return y;


    }


    /**
     * Back propagation in the given input
     * @param y
     * @return
     */
    public LSTMGradient backward(INDArray y) {
        INDArray dY = Nd4j.vstack(Nd4j.zeros(y.columns()),y);
        INDArray dWd = hOut.transpose().mmul(dY);
        INDArray dBd = Nd4j.sum(dWd,0);
        INDArray dHout = dY.mmul(decoderWeights);
        if(conf.getDropOut() > 0) {
            dHout.muli(u2);
        }



        INDArray dIFog = Nd4j.zeros(iFog.shape());
        INDArray dIFogF = Nd4j.zeros(iFogF.shape());
        INDArray dRecurrentWeights = Nd4j.zeros(recurrentWeights.shape());
        INDArray dHin = Nd4j.zeros(hIn.shape());

        INDArray dC = Nd4j.zeros(c.shape());
        INDArray dx = Nd4j.zeros(x.shape());
        int n = x.rows();
        int d = decoderWeights.rows();


        for(int t = n -1; t > 0; t--) {
            if(conf.getActivationFunction().type().equals("tanh")) {
                INDArray tanhCt = tanh(c.slice(t));
                dIFogF.slice(t).get(interval(2 * d,3 * d)).assign(tanhCt.mul(dHout.slice(t)));
                dC.slice(t).addi(pow(tanhCt,2).rsubi(1).muli(iFogF.slice(t).get(interval(2 * d, 3 * d)).mul(dHout.slice(t))));
            }
            else {
                dIFogF.slice(t).get(interval(2 * d,3 * d)).assign(c.slice(t).mul(dHout.slice(t)));
                dC.slice(t).addi(iFogF.slice(t).get(interval(2 * d,3 * d)).mul(dHout.slice(t)));
            }

            if(t > 0) {
                dIFogF.slice(t).get(interval(d, 2 * d)).assign(c.slice(t - 1).mul(dC.slice(t)));
                dC.slice(t - 1).addi(iFogF.slice(t).get(interval(d,2 * d)).mul(dC.slice(t)));
            }

            dIFogF.slice(t).get(interval(0,d)).assign(iFogF.slice(t).get(interval(3 * d,iFogF.columns())).mul(dC.slice(t)));
            dIFogF.slice(t).get(interval(3 * d,dIFogF.columns())).assign(iFogF.slice(t).get(interval(0,d)).mul(dC.slice(t)));

            dIFog.slice(t).get(interval(3 * d,dIFog.columns())).assign(pow(iFogF.slice(t).get(interval(3 * d,iFogF.columns())),2).rsubi(1).mul(dIFogF.slice(t).get(interval(3 * d,dIFogF.columns()))));
            y = iFogF.slice(t).get(interval(0,3 * d));
            dIFogF.slice(t).get(interval(0, 3 * d)).assign(y.mul(y.rsub(1)).mul(dIFogF.slice(t).get(interval(0, 3 * d))));

            dRecurrentWeights.addi(hIn.slice(t).mmul(dIFogF.slice(t)));
            dHin.slice(t).assign(dIFog.slice(t).mmul(recurrentWeights.transpose()));

            dx.slice(t).assign(dHin.slice(t).get(interval(1, 1 + d)));

            if(t > 0) {
                dHout.slice(t - 1).addi(dHin.slice(t).get(interval(1 + d, dHin.columns())));

            }

            if(conf.getDropOut() > 0) {
                dx.muli(u);
            }
        }


        clear();

        return new LSTMGradient(dY,dWd,dBd,dHout,dIFog,dIFogF,dRecurrentWeights,dHin,dC,dx);

    }






    /**
     * Prediction with beam search
     * @param xi
     * @param ws
     * @return
     */
    public Collection<Pair<List<Integer>,Double>> predict(INDArray xi,INDArray ws) {
        int d = decoderWeights.rows();
        Triple<INDArray,INDArray,INDArray> yhc = lstmTick(xi,Nd4j.zeros(d),Nd4j.zeros(d));
        BeamSearch search = new BeamSearch(20,ws,yhc.getSecond(),yhc.getThird());
        Collection<Pair<List<Integer>,Double>> ret = search.search();
        return ret;

    }



    private void clear() {
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


    public static class LSTMGradient implements Gradient {
        INDArray dY;
        INDArray dWd;
        INDArray dBd;
        INDArray dHout;
        INDArray dIFog;
        INDArray dIFogF;
        INDArray dRecurrentWeights;
        INDArray dHin;
        INDArray dC;
        INDArray dx;

        public LSTMGradient(INDArray dY, INDArray dWd, INDArray dBd, INDArray dHout, INDArray dIFog, INDArray dIFogF, INDArray dRecurrentWeights, INDArray dHin, INDArray dC, INDArray dx) {
            this.dY = dY;
            this.dWd = dWd;
            this.dBd = dBd;
            this.dHout = dHout;
            this.dIFog = dIFog;
            this.dIFogF = dIFogF;
            this.dRecurrentWeights = dRecurrentWeights;
            this.dHin = dHin;
            this.dC = dC;
            this.dx = dx;
        }

        public INDArray getdY() {
            return dY;
        }

        public void setdY(INDArray dY) {
            this.dY = dY;
        }

        public INDArray getdWd() {
            return dWd;
        }

        public void setdWd(INDArray dWd) {
            this.dWd = dWd;
        }

        public INDArray getdBd() {
            return dBd;
        }

        public void setdBd(INDArray dBd) {
            this.dBd = dBd;
        }

        public INDArray getdHout() {
            return dHout;
        }

        public void setdHout(INDArray dHout) {
            this.dHout = dHout;
        }

        public INDArray getdIFog() {
            return dIFog;
        }

        public void setdIFog(INDArray dIFog) {
            this.dIFog = dIFog;
        }

        public INDArray getdIFogF() {
            return dIFogF;
        }

        public void setdIFogF(INDArray dIFogF) {
            this.dIFogF = dIFogF;
        }

        public INDArray getdRecurrentWeights() {
            return dRecurrentWeights;
        }

        public void setdRecurrentWeights(INDArray dRecurrentWeights) {
            this.dRecurrentWeights = dRecurrentWeights;
        }

        public INDArray getdHin() {
            return dHin;
        }

        public void setdHin(INDArray dHin) {
            this.dHin = dHin;
        }

        public INDArray getdC() {
            return dC;
        }

        public void setdC(INDArray dC) {
            this.dC = dC;
        }

        public INDArray getDx() {
            return dx;
        }

        public void setDx(INDArray dx) {
            this.dx = dx;
        }

        @Override
        public INDArray gradient() {
            return Nd4j.concat(0,dY.linearView(),dWd.linearView(),dBd.linearView(), dHout.linearView(),dIFog.linearView(),dIFogF.linearView(),dHin.linearView(),dRecurrentWeights.linearView(),dHin.linearView(),dC.linearView(),dx.linearView());
        }

        @Override
        public void clear() {

        }
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
    public double score() {
        INDArray forward = Activations.softMaxRows().apply(forward(xi, xs));
        return -log(forward.sum(Integer.MAX_VALUE)).getDouble(0);
    }

    @Override
    public INDArray transform(INDArray data) {
        return Activations.softMaxRows().apply(forward(xi,xs));
    }

    @Override
    public INDArray params() {
        return Nd4j.concat(0,recurrentWeights.linearView(),decoderWeights.linearView(),decoderBias.linearView());
    }

    @Override
    public int numParams() {
        return recurrentWeights.length() + decoderWeights.length() + decoderBias.length();
    }

    @Override
    public void setParams(INDArray params) {
       int count = 0;
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
         Solver solver = new Solver.Builder()
                 .configure(conf).model(this)
                 .listeners(conf.getListeners()).build();
        solver.optimize();
    }

    @Override
    public void iterate(INDArray input) {

    }

    @Override
    public Gradient getGradient() {
        INDArray forward = forward(xi,xs);
        INDArray probas = Activations.softmax().applyDerivative(forward);
        return backward(probas);
    }

    @Override
    public Pair<Gradient, Double> gradientAndScore() {
        return new Pair<>(getGradient(),score());
    }

    @Override
    public int batchSize() {
        return xi.rows();
    }


    public static class Builder {
        private NeuralNetConfiguration conf;


        public Builder configure(NeuralNetConfiguration conf) {
            this.conf = conf;
            return this;
        }


        public LSTM build() {
            LSTM ret = new LSTM();
            ret.conf = conf;
            ret.init();
            return ret;
        }

    }

}
