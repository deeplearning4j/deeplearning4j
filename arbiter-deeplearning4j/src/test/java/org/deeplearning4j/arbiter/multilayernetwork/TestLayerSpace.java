/*-
 *
 *  * Copyright 2016 Skymind,Inc.
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
package org.deeplearning4j.arbiter.multilayernetwork;

import org.apache.commons.lang3.ArrayUtils;
import org.deeplearning4j.arbiter.layers.*;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.continuous.ContinuousParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.discrete.DiscreteParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.integer.IntegerParameterSpace;
import org.deeplearning4j.nn.conf.layers.*;
import org.junit.Test;

import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class TestLayerSpace {

    @Test
    public void testBasic1(){

        DenseLayer expected = new DenseLayer.Builder()
                .nOut(13).activation("relu").build();

        DenseLayerSpace space = new DenseLayerSpace.Builder()
                .nOut(13).activation("relu").build();

        int nParam = space.numParameters();
        assertEquals(0,nParam);
        DenseLayer actual = space.getValue(new double[nParam]);

        assertEquals(expected,actual);
    }

    @Test
    public void testBasic2(){

        String[] actFns = new String[]{"softsign","relu","leakyrelu"};
        Random r = new Random(12345);

        for( int i=0; i<20; i++ ) {

            new DenseLayer.Builder().build();

            DenseLayerSpace ls = new DenseLayerSpace.Builder()
                    .nOut(20)
                    .learningRate(new ContinuousParameterSpace(0.3,0.4))
                    .l2(new ContinuousParameterSpace(0.01,0.1))
                    .activation(new DiscreteParameterSpace<>(actFns))
                    .build();

            //Set the parameter numbers...
            List<ParameterSpace> list = ls.collectLeaves();
            for( int j=0; j<list.size(); j++ ){
                list.get(j).setIndices(j);
            }

            int nParam = ls.numParameters();
            assertEquals(3,nParam);

            double[] d = new double[nParam];
            for( int j=0; j<d.length; j++ ){
                d[j] = r.nextDouble();
            }

            DenseLayer l = ls.getValue(d);

            assertEquals(20, l.getNOut());
            double lr = l.getLearningRate();
            double l2 = l.getL2();
            String activation = l.getActivationFn().toString();

            System.out.println(lr + "\t" + l2 + "\t" + activation);

            assertTrue(lr >= 0.3 && lr <= 0.4);
            assertTrue(l2 >= 0.01 && l2 <= 0.1 );
            assertTrue(containsActivationFunction(actFns, activation));
        }
    }

    @Test
    public void testBatchNorm(){
        BatchNormalizationSpace sp = new BatchNormalizationSpace.Builder()
                .gamma(1.5)
                .beta(new ContinuousParameterSpace(2,3))
                .lockGammaBeta(true)
                .build();

        //Set the parameter numbers...
        List<ParameterSpace> list = sp.collectLeaves();
        for( int j=0; j<list.size(); j++ ){
            list.get(j).setIndices(j);
        }

        BatchNormalization bn = sp.getValue(new double[]{0.6});
        assertTrue(bn.isLockGammaBeta());
        assertEquals(1.5, bn.getGamma(), 0.0);
        assertEquals(0.6*(3-2)+2, bn.getBeta(), 1e-4);
    }

    @Test
    public void testActivationLayer(){

        String[] actFns = new String[]{"softsign","relu","leakyrelu"};

        ActivationLayerSpace als = new ActivationLayerSpace.Builder()
                .activation(new DiscreteParameterSpace<>(actFns))
                .build();
        //Set the parameter numbers...
        List<ParameterSpace> list = als.collectLeaves();
        for( int j=0; j<list.size(); j++ ){
            list.get(j).setIndices(j);
        }

        int nParam = als.numParameters();
        assertEquals(1, nParam);

        Random r = new Random(12345);

        for( int i=0; i<20; i++ ) {

            double[] d = new double[nParam];
            for( int j=0; j<d.length; j++ ){
                d[j] = r.nextDouble();
            }

            ActivationLayer al = als.getValue(d);
            String activation = al.getActivationFn().toString();

            System.out.println(activation);

            assertTrue(containsActivationFunction(actFns, activation));
        }
    }

    @Test
    public void testEmbeddingLayer(){

        String[] actFns = new String[]{"softsign","relu","leakyrelu"};

        EmbeddingLayerSpace els = new EmbeddingLayerSpace.Builder()
                .activation(new DiscreteParameterSpace<>(actFns))
                .nIn(10).nOut(new IntegerParameterSpace(10,20))
                .build();
        //Set the parameter numbers...
        List<ParameterSpace> list = els.collectLeaves();
        for( int j=0; j<list.size(); j++ ){
            list.get(j).setIndices(j);
        }

        int nParam = els.numParameters();
        assertEquals(2, nParam);

        Random r = new Random(12345);

        for( int i=0; i<20; i++ ) {

            double[] d = new double[nParam];
            for( int j=0; j<d.length; j++ ){
                d[j] = r.nextDouble();
            }

            EmbeddingLayer el = els.getValue(d);
            String activation = el.getActivationFn().toString();
            int nOut = el.getNOut();

            System.out.println(activation + "\t" + nOut);

            assertTrue(containsActivationFunction(actFns, activation));
            assertTrue(nOut >= 10 && nOut <= 20);
        }
    }

    @Test
    public void testGravesBidirectionalLayer(){

        String[] actFns = new String[]{"softsign","relu","leakyrelu"};

        GravesBidirectionalLSTMLayerSpace ls = new GravesBidirectionalLSTMLayerSpace.Builder()
                .activation(new DiscreteParameterSpace<>(actFns))
                .forgetGateBiasInit(new ContinuousParameterSpace(0.5,0.8))
                .nIn(10).nOut(new IntegerParameterSpace(10,20))
                .build();
        //Set the parameter numbers...
        List<ParameterSpace> list = ls.collectLeaves();
        for( int j=0; j<list.size(); j++ ){
            list.get(j).setIndices(j);
        }

        int nParam = ls.numParameters();
        assertEquals(3, nParam);

        Random r = new Random(12345);

        for( int i=0; i<20; i++ ) {

            double[] d = new double[nParam];
            for( int j=0; j<d.length; j++ ){
                d[j] = r.nextDouble();
            }

            GravesBidirectionalLSTM el = ls.getValue(d);
            String activation = el.getActivationFn().toString();
            int nOut = el.getNOut();
            double forgetGate = el.getForgetGateBiasInit();

            System.out.println(activation + "\t" + nOut + "\t" + forgetGate);

            assertTrue(containsActivationFunction(actFns, activation));
            assertTrue(nOut >= 10 && nOut <= 20);
            assertTrue(forgetGate >= 0.5 && forgetGate <= 0.8);
        }
    }

    private static boolean containsActivationFunction(String[] activationFunctions, String activationFunction) {
        for (String af : activationFunctions) {
            if (activationFunction.startsWith(af)) return true;
        }
        return false;
    }
}
