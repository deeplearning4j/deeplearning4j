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
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Random;

import static org.junit.Assert.assertEquals;

public class UpdaterTest extends BaseNd4jTest {

    private static final Logger log = LoggerFactory.getLogger(UpdaterTest.class);

    public UpdaterTest(String name, Nd4jBackend backend) {
        super(name, backend);
    }

    public UpdaterTest() {
    }

    public UpdaterTest(Nd4jBackend backend) {
        super(backend);
    }

    public UpdaterTest(String name) {
        super(name);
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
        int n = 5;
        Nd4j.getRandom().setSeed(12345);
        Random r = new Random(12345);
        double[] lrs = new double[n];
        AdaGrad[] adaGrads = new AdaGrad[n];
        AdaGrad[] adaGradsExFirst = new AdaGrad[n-1];
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

            if(i>0) adaGradsExFirst[i-1] = adaGrads[i];
        }
        avgLr /= n;
        avgState.divi(n);

        adaGrads[0].combineUpdaters(adaGradsExFirst);

        double lrCombined = adaGrads[0].getLearningRate();
        INDArray histCombined = adaGrads[0].getHistoricalGradient();

        assertEquals(avgLr,lrCombined,1e-10);
        assertEquals(avgState,histCombined);
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
        int n = 5;
        Nd4j.getRandom().setSeed(12345);
        Random r = new Random(12345);
        double[] lrs = new double[n];
        double[] momentums = new double[n];
        Nesterovs[] nesterovs = new Nesterovs[n];
        Nesterovs[] nesterovsExFirst = new Nesterovs[n-1];
        INDArray[] vs = new INDArray[n];

        double avgLr = 0.0;
        double avgMomentums = 0.0;
        INDArray avgState = Nd4j.zeros(1,10);
        for( int i=0; i<vs.length; i++ ){
            lrs[i] = r.nextDouble();
            momentums[i] = r.nextDouble();
            avgLr += lrs[i];
            avgMomentums += momentums[i];
            nesterovs[i] = new Nesterovs(momentums[i],lrs[i]);
            vs[i] = Nd4j.rand(1, 10);
            avgState.addi(vs[i]);
            nesterovs[i].setV(vs[i].dup());

            if(i>0) nesterovsExFirst[i-1] = nesterovs[i];
        }
        avgLr /= n;
        avgMomentums /= n;
        avgState.divi(n);

        nesterovs[0].combineUpdaters(nesterovsExFirst);

        double lrCombined = nesterovs[0].getLearningRate();
        double momentumCombined = nesterovs[0].getMomentum();
        INDArray vCombined = nesterovs[0].getV();

        assertEquals(avgLr,lrCombined,1e-10);
        assertEquals(avgMomentums,momentumCombined,1e-10);
        assertEquals(avgState,vCombined);
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
        int n = 5;
        Nd4j.getRandom().setSeed(12345);
        Random r = new Random(12345);
        double[] rhos = new double[n];
        AdaDelta[] adaDeltas = new AdaDelta[n];
        AdaDelta[] adaDeltasExFirst = new AdaDelta[n-1];
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

            if(i>0) adaDeltasExFirst[i-1] = adaDeltas[i];
        }
        avgRho /= n;
        avgStateMsg.divi(n);
        avgStateMsdxs.divi(n);

        adaDeltas[0].combineUpdaters(adaDeltasExFirst);

        double rhoCombined = adaDeltas[0].getRho();
        INDArray msgsCombined = adaDeltas[0].getMsg();
        INDArray msdxsCombined = adaDeltas[0].getMsdx();

        assertEquals(avgRho,rhoCombined,1e-10);
        assertEquals(avgStateMsg,msgsCombined);
        assertEquals(avgStateMsdxs,msdxsCombined);
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
        int n = 5;
        Nd4j.getRandom().setSeed(12345);
        Random r = new Random(12345);
        double[] lrs = new double[n];
        Adam[] adams = new Adam[n];
        Adam[] adamsExFirst = new Adam[n-1];
        INDArray[] ms = new INDArray[n];
        INDArray[] vs = new INDArray[n];

        double avgLr = 0.0;
        INDArray avgStateM = Nd4j.zeros(1,10);
        INDArray avgStateV = Nd4j.zeros(1,10);
        for( int i=0; i<ms.length; i++ ){
            lrs[i] = r.nextDouble();
            avgLr += lrs[i];
            adams[i] = new Adam(lrs[i]);
            ms[i] = Nd4j.rand(1, 10);
            vs[i] = Nd4j.rand(1, 10);
            avgStateM.addi(ms[i]);
            avgStateV.addi(vs[i]);
            adams[i].setM(ms[i].dup());
            adams[i].setV(vs[i].dup());

            if(i>0) adamsExFirst[i-1] = adams[i];
        }
        avgLr /= n;
        avgStateM.divi(n);
        avgStateV.divi(n);

        adams[0].combineUpdaters(adamsExFirst);

        double lrCombined = adams[0].getLearningRate();
        INDArray msgsCombined = adams[0].getM();
        INDArray msdxsCombined = adams[0].getV();

        assertEquals(avgLr,lrCombined,1e-10);
        assertEquals(avgStateM,msgsCombined);
        assertEquals(avgStateV,msdxsCombined);
    }

    @Test
    public void testRmsPropCombining() {
        int n = 5;
        Nd4j.getRandom().setSeed(12345);
        Random r = new Random(12345);
        double[] lrs = new double[n];
        double[] rmsDecays = new double[n];
        RmsProp[] rmsProps = new RmsProp[n];
        RmsProp[] rmsPropsExFirst = new RmsProp[n-1];
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

            if(i>0) rmsPropsExFirst[i-1] = rmsProps[i];
        }
        avgLr /= n;
        avgRmsDecay /= n;
        avgLastGradient.divi(n);

        rmsProps[0].combineUpdaters(rmsPropsExFirst);

        double lrCombined = rmsProps[0].getLearningRate();
        double rmsDecayCombined = rmsProps[0].getRmsDecay();
        INDArray lastGradientCombined = rmsProps[0].getLastGradient();

        assertEquals(avgLr,lrCombined,1e-10);
        assertEquals(avgRmsDecay,rmsDecayCombined,1e-10);
        assertEquals(avgLastGradient,lastGradientCombined);
    }

    @Test
    public void testSgdCombining() {
        int n = 5;
        Nd4j.getRandom().setSeed(12345);
        Random r = new Random(12345);
        double[] lrs = new double[n];
        Sgd[] sgds = new Sgd[n];
        Sgd[] sgdsExFirst = new Sgd[n-1];

        double avgLr = 0.0;
        for( int i=0; i<n; i++ ){
            lrs[i] = r.nextDouble();
            avgLr += lrs[i];
            sgds[i] = new Sgd(lrs[i]);

            if(i>0) sgdsExFirst[i-1] = sgds[i];
        }
        avgLr /= n;

        sgds[0].combineUpdaters(sgdsExFirst);

        double lrCombined = sgds[0].getLearningRate();

        assertEquals(avgLr,lrCombined,1e-10);
    }



    @Override
    public char ordering() {
        return 'f';
    }
}
