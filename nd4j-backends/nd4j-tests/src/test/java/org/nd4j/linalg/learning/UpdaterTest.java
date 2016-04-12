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

package org.nd4j.linalg.learning;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Random;

import static org.junit.Assert.assertEquals;

@RunWith(Parameterized.class)
public class UpdaterTest extends BaseNd4jTest {

    private static final Logger log = LoggerFactory.getLogger(UpdaterTest.class);



    public UpdaterTest(Nd4jBackend backend) {
        super(backend);
    }


    @Test
    public void testAdaGrad1() {
        int rows = 1;
        int cols = 1;


        AdaGrad grad = new AdaGrad(rows, cols, 1e-3);
        INDArray W = Nd4j.ones(rows, cols);
        assertEquals(1e-1, grad.getGradient(W, 0).getDouble(0), 1e-1);
    }

    @Test
    public void testAdaGradCombining() {
        int n = 7;
        Nd4j.getRandom().setSeed(12345);
        Random r = new Random(12345);
        double[] lrs = new double[n];
        AdaGrad[] adaGrads = new AdaGrad[n];
        INDArray[] arr = new INDArray[n];

        double avgLr = 0.0;
        INDArray avgState = Nd4j.zeros(1,10);
        for( int i=0; i<arr.length; i++ ){
            lrs[i] = r.nextDouble();
            avgLr += lrs[i];
            adaGrads[i] = new AdaGrad(lrs[i]);
            arr[i] = Nd4j.rand(1, 10);
            avgState.addi(arr[i]);
            adaGrads[i].setHistoricalGradient(arr[i].dup());
        }
        avgLr /= n;
        avgState.divi(n);

        GradientUpdaterAggregator ag = adaGrads[0].getAggregator(true);
        for( int i=1; i<n; i++ ){
            ag.aggregate(adaGrads[i]);
        }

        AdaGrad combined = (AdaGrad)ag.getUpdater();

        double lrCombined = combined.getLearningRate();
        INDArray histCombined = combined.getHistoricalGradient();

        assertEquals(avgLr,lrCombined,1e-10);
        assertEquals(avgState,histCombined);

        //Check merging of AdaGradAggregators:
        GradientUpdaterAggregator first = adaGrads[0].getAggregator(false);
        GradientUpdaterAggregator second = adaGrads[2].getAggregator(false);
        for(int i=0; i<n; i++ ){
            if(i<2){
                first.aggregate(adaGrads[i]);
            } else {
                second.aggregate(adaGrads[i]);
            }
        }

        GradientUpdaterAggregator agMerged = first.combine(second);
        AdaGrad combined2 = (AdaGrad) agMerged.getUpdater();
        assertEquals(avgLr,combined2.getLearningRate(),1e-10);
        assertEquals(avgState,combined2.getHistoricalGradient());
    }

    @Test
    public void testNesterovs() {
        int rows = 10;
        int cols = 2;


        Nesterovs grad = new Nesterovs(0.5);
        INDArray W = Nd4j.zeros(rows, cols);
        Distribution dist = Nd4j.getDistributions().createNormal(1, 1);
        for (int i = 0; i < W.rows(); i++)
            W.putRow(i, Nd4j.create(dist.sample(W.columns())));

        for (int i = 0; i < 5; i++) {
            String learningRates = String.valueOf("\nAdagrad\n " + grad.getGradient(W, i)).replaceAll(";", "\n");
            System.out.println(learningRates);
            W.addi(Nd4j.randn(rows, cols));
        }
    }

    @Test
    public void testNesterovCombining() {
        int n = 7;
        Nd4j.getRandom().setSeed(12345);
        Random r = new Random(12345);
        double[] lrs = new double[n];
        double[] momentums = new double[n];
        Nesterovs[] nesterovs = new Nesterovs[n];
        INDArray[] vs = new INDArray[n];

        double avgLr = 0.0;
        double avgMomentums = 0.0;
        INDArray avgState = Nd4j.zeros(1,10);
        for( int i = 0; i < vs.length; i++){
            lrs[i] = r.nextDouble();
            momentums[i] = r.nextDouble();
            avgLr += lrs[i];
            avgMomentums += momentums[i];
            nesterovs[i] = new Nesterovs(momentums[i],lrs[i]);
            vs[i] = Nd4j.rand(1, 10);
            avgState.addi(vs[i]);
            nesterovs[i].setV(vs[i].dup());
        }
        avgLr /= n;
        avgMomentums /= n;
        avgState.divi(n);

        GradientUpdaterAggregator ag = nesterovs[0].getAggregator(true);
        for( int i=1; i<n; i++) ag.aggregate(nesterovs[i]);

        Nesterovs combined = (Nesterovs)ag.getUpdater();

        assertEquals(avgLr,combined.getLearningRate(),1e-10);
        assertEquals(avgMomentums,combined.getMomentum(),1e-10);
        assertEquals(avgState,combined.getV());

        //Check merging of NesterovsAggregators:
        GradientUpdaterAggregator first = nesterovs[0].getAggregator(false);
        GradientUpdaterAggregator second = nesterovs[2].getAggregator(false);
        for(int i=0; i < n; i++ ){
            if(i < 2){
                first.aggregate(nesterovs[i]);
            } else {
                second.aggregate(nesterovs[i]);
            }
        }

        GradientUpdaterAggregator agMerged = first.combine(second);
        Nesterovs combined2 = (Nesterovs) agMerged.getUpdater();
        assertEquals(avgLr,combined2.getLearningRate(),1e-10);
        assertEquals(avgMomentums,combined2.getMomentum(),1e-10);
        assertEquals(avgState,combined2.getV());
    }


    @Test
    public void testAdaGrad() {
        int rows = 10;
        int cols = 2;


        AdaGrad grad = new AdaGrad(rows, cols, 0.1);
        INDArray W = Nd4j.zeros(rows, cols);
        Distribution dist = Nd4j.getDistributions().createNormal(1, 1);
        for (int i = 0; i < W.rows(); i++)
            W.putRow(i, Nd4j.create(dist.sample(W.columns())));

        for (int i = 0; i < 5; i++) {
            String learningRates = String.valueOf("\nAdagrad\n " + grad.getGradient(W, i)).replaceAll(";", "\n");
            System.out.println(learningRates);
            W.addi(Nd4j.randn(rows, cols));
        }

    }

    @Test
    public void testAdaDelta() {
        int rows = 10;
        int cols = 2;


        AdaDelta grad = new AdaDelta();
        INDArray W = Nd4j.zeros(rows, cols);
        Distribution dist = Nd4j.getDistributions().createNormal(1e-3, 1e-3);
        for (int i = 0; i < W.rows(); i++)
            W.putRow(i, Nd4j.create(dist.sample(W.columns())));

        for (int i = 0; i < 5; i++) {
            String learningRates = String.valueOf("\nAdaelta\n " + grad.getGradient(W, i)).replaceAll(";", "\n");
            System.out.println(learningRates);
            W.addi(Nd4j.randn(rows, cols));
        }
    }

    @Test
    public void testAdaDeltaCombining() {
        int n = 7;
        Nd4j.getRandom().setSeed(12345);
        Random r = new Random(12345);
        double[] rhos = new double[n];
        AdaDelta[] adaDeltas = new AdaDelta[n];
        INDArray[] msgs = new INDArray[n];
        INDArray[] msdxs = new INDArray[n];

        double avgRho = 0.0;
        INDArray avgStateMsg = Nd4j.zeros(1,10);
        INDArray avgStateMsdxs = Nd4j.zeros(1,10);
        for( int i=0; i<msgs.length; i++ ){
            rhos[i] = r.nextDouble();
            avgRho += rhos[i];
            adaDeltas[i] = new AdaDelta(rhos[i]);
            msgs[i] = Nd4j.rand(1, 10);
            msdxs[i] = Nd4j.rand(1, 10);
            avgStateMsg.addi(msgs[i]);
            avgStateMsdxs.addi(msdxs[i]);
            adaDeltas[i].setMsg(msgs[i].dup());
            adaDeltas[i].setMsdx(msdxs[i].dup());
        }
        avgRho /= n;
        avgStateMsg.divi(n);
        avgStateMsdxs.divi(n);

        GradientUpdaterAggregator ag = adaDeltas[0].getAggregator(true);
        for( int i=1; i<n; i++ ) ag.aggregate(adaDeltas[i]);

        AdaDelta combined = (AdaDelta)ag.getUpdater();

        assertEquals(avgRho,combined.getRho(),1e-10);
        assertEquals(avgStateMsg,combined.getMsg());
        assertEquals(avgStateMsdxs,combined.getMsdx());

        //Check merging of AdaDelta:
        GradientUpdaterAggregator first = adaDeltas[0].getAggregator(false);
        GradientUpdaterAggregator second = adaDeltas[2].getAggregator(false);
        for(int i = 0; i < n; i++ ){
            if(i < 2){
                first.aggregate(adaDeltas[i]);
            } else {
                second.aggregate(adaDeltas[i]);
            }
        }

        GradientUpdaterAggregator agMerged = first.combine(second);
        AdaDelta combined2 = (AdaDelta) agMerged.getUpdater();
        assertEquals(avgRho,combined2.getRho(),1e-10);
        assertEquals(avgStateMsg,combined2.getMsg());
        assertEquals(avgStateMsdxs,combined2.getMsdx());
    }

    @Test
    public void testAdam() {
        int rows = 10;
        int cols = 2;


        Adam grad = new Adam();
        INDArray W = Nd4j.zeros(rows, cols);
        Distribution dist = Nd4j.getDistributions().createNormal(1e-3, 1e-3);
        for (int i = 0; i < W.rows(); i++)
            W.putRow(i, Nd4j.create(dist.sample(W.columns())));

        for (int i = 0; i < 5; i++) {
            String learningRates = String.valueOf("\nAdam\n " + grad.getGradient(W, i)).replaceAll(";", "\n");
            System.out.println(learningRates);
            W.addi(Nd4j.randn(rows, cols));
        }
    }

    @Test
    public void testAdamCombining() {
        int n = 7;
        Nd4j.getRandom().setSeed(12345);
        Random r = new Random(12345);
        double[] lrs = new double[n];
        double[] beta1s = new double[n];
        double[] beta2s = new double[n];
        double[] eps = new double[n];
        Adam[] adams = new Adam[n];
        INDArray[] ms = new INDArray[n];
        INDArray[] vs = new INDArray[n];

        double avgLr = 0.0;
        double avgBeta1 = 0.0;
        double avgBeta2 = 0.0;
        double avgEps = 0.0;
        INDArray avgStateM = Nd4j.zeros(1,10);
        INDArray avgStateV = Nd4j.zeros(1,10);
        for(int i = 0;  i < n; i++) {
            lrs[i] = r.nextDouble();
            beta1s[i] = r.nextDouble();
            beta2s[i] = r.nextDouble();
            eps[i] = r.nextDouble();
            avgLr += lrs[i];
            avgBeta1 += beta1s[i];
            avgBeta2 += beta2s[i];
            avgEps += eps[i];
            adams[i] = new Adam(lrs[i]);
            adams[i].setBeta1(beta1s[i]);
            adams[i].setBeta2(beta2s[i]);
            adams[i].setEpsilon(eps[i]);
            ms[i] = Nd4j.rand(1, 10);
            vs[i] = Nd4j.rand(1, 10);
            avgStateM.addi(ms[i]);
            avgStateV.addi(vs[i]);
            adams[i].setM(ms[i].dup());
            adams[i].setV(vs[i].dup());
        }
        avgLr /= n;
        avgBeta1 /= n;
        avgBeta2 /= n;
        avgEps /= n;
        avgStateM.divi(n);
        avgStateV.divi(n);

        GradientUpdaterAggregator ag = adams[0].getAggregator(true);
        for( int i=1; i<n; i++) ag.aggregate(adams[i]);

        Adam combined = (Adam)ag.getUpdater();

        assertEquals(avgLr,combined.getLearningRate(),1e-10);
        assertEquals(avgBeta1,combined.getBeta1(),1e-10);
        assertEquals(avgBeta2,combined.getBeta2(),1e-10);
        assertEquals(avgEps,combined.getEpsilon(),1e-10);
        assertEquals(avgStateM,combined.getM());
        assertEquals(avgStateV,combined.getV());

        //Check merging of AdamAggregators:
        GradientUpdaterAggregator first = adams[0].getAggregator(false);
        GradientUpdaterAggregator second = adams[2].getAggregator(false);
        for(int i=0; i < n; i++ ){
            if(i < 2){
                first.aggregate(adams[i]);
            } else {
                second.aggregate(adams[i]);
            }
        }
        GradientUpdaterAggregator agMerged = first.combine(second);
        Adam combined2 = (Adam) agMerged.getUpdater();
        assertEquals(avgLr,combined2.getLearningRate(),1e-10);
        assertEquals(avgBeta1,combined2.getBeta1(),1e-10);
        assertEquals(avgBeta2,combined2.getBeta2(),1e-10);
        assertEquals(avgEps,combined2.getEpsilon(),1e-10);
        assertEquals(avgStateM,combined2.getM());
        assertEquals(avgStateV,combined2.getV());
    }

    @Test
    public void testRmsPropCombining() {
        int n = 7;
        Nd4j.getRandom().setSeed(12345);
        Random r = new Random(12345);
        double[] lrs = new double[n];
        double[] rmsDecays = new double[n];
        RmsProp[] rmsProps = new RmsProp[n];
        INDArray[] lastGradients = new INDArray[n];

        double avgLr = 0.0;
        double avgRmsDecay = 0.0;
        INDArray avgLastGradient = Nd4j.zeros(1,10);
        for( int i=0; i<lastGradients.length; i++ ){
            lrs[i] = r.nextDouble();
            rmsDecays[i] = r.nextDouble();
            avgLr += lrs[i];
            avgRmsDecay += rmsDecays[i];
            rmsProps[i] = new RmsProp(lrs[i],rmsDecays[i]);
            lastGradients[i] = Nd4j.rand(1, 10);
            avgLastGradient.addi(lastGradients[i]);
            rmsProps[i].setLastGradient(lastGradients[i].dup());
        }
        avgLr /= n;
        avgRmsDecay /= n;
        avgLastGradient.divi(n);

        GradientUpdaterAggregator ag = rmsProps[0].getAggregator(true);
        for( int i=1; i<n; i++) ag.aggregate(rmsProps[i]);

        RmsProp combined = (RmsProp)ag.getUpdater();

        assertEquals(avgLr,combined.getLearningRate(),1e-10);
        assertEquals(avgRmsDecay,combined.getRmsDecay(),1e-10);
        assertEquals(avgLastGradient,combined.getLastGradient());

        //Check merging of RmsPropAggregators:
        GradientUpdaterAggregator first = rmsProps[0].getAggregator(false);
        GradientUpdaterAggregator second = rmsProps[2].getAggregator(false);
        for(int i=0; i<n; i++ ){
            if(i<2){
                first.aggregate(rmsProps[i]);
            } else {
                second.aggregate(rmsProps[i]);
            }
        }
        GradientUpdaterAggregator agMerged = first.combine(second);
        RmsProp combined2 = (RmsProp) agMerged.getUpdater();
        assertEquals(avgLr,combined2.getLearningRate(),1e-10);
        assertEquals(avgRmsDecay,combined2.getRmsDecay(),1e-10);
        assertEquals(avgLastGradient,combined2.getLastGradient());
    }

    @Test
    public void testSgdCombining() {
        int n = 7;
        Nd4j.getRandom().setSeed(12345);
        Random r = new Random(12345);
        double[] lrs = new double[n];
        Sgd[] sgds = new Sgd[n];

        double avgLr = 0.0;
        for( int i=0; i<n; i++ ){
            lrs[i] = r.nextDouble();
            avgLr += lrs[i];
            sgds[i] = new Sgd(lrs[i]);
        }
        avgLr /= n;

        GradientUpdaterAggregator ag = sgds[0].getAggregator(true);
        for( int i=1; i<n; i++) ag.aggregate(sgds[i]);

        Sgd combined = (Sgd)ag.getUpdater();

        assertEquals(avgLr,combined.getLearningRate(),1e-10);

        //Check merging of SgdAggregators:
        GradientUpdaterAggregator first = sgds[0].getAggregator(false);
        GradientUpdaterAggregator second = sgds[2].getAggregator(false);
        for(int i=0; i<n; i++ ){
            if(i<2){
                first.aggregate(sgds[i]);
            } else {
                second.aggregate(sgds[i]);
            }
        }
        GradientUpdaterAggregator agMerged = first.combine(second);
        Sgd combined2 = (Sgd) agMerged.getUpdater();
        assertEquals(avgLr,combined2.getLearningRate(),1e-10);
    }



    @Override
    public char ordering() {
        return 'f';
    }
}
