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
 *
 */

package org.nd4j.linalg;


import org.apache.commons.math3.linear.BlockRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.util.Pair;
import org.junit.After;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.checkutil.NDArrayCreationUtil;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.checkutil.CheckUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**Tests comparing Nd4j ops to other libraries
 */
public  class Nd4jTestsComparisonC extends BaseNd4jTest {
    private static Logger log = LoggerFactory.getLogger(Nd4jTestsComparisonC.class);

    public static final int SEED = 123;

    public Nd4jTestsComparisonC() {
    }

    public Nd4jTestsComparisonC(String name) {
        super(name);
    }

    public Nd4jTestsComparisonC(Nd4jBackend backend) {
        super(backend);
    }

    public Nd4jTestsComparisonC(String name, Nd4jBackend backend) {
        super(name, backend);
    }

    @Before
    public void before() {
        super.before();
        Nd4j.factory().setDType(DataBuffer.Type.DOUBLE);
        Nd4j.dtype = DataBuffer.Type.DOUBLE;
        Nd4j.getRandom().setSeed(SEED);

    }

    @After
    public void after() {
        super.after();
        Nd4j.factory().setDType(DataBuffer.Type.DOUBLE);
        Nd4j.dtype = DataBuffer.Type.DOUBLE;
    }

    @Override
    public char ordering() {
        return 'c';
    }


    @Test
    public void testGemmWithOpsCommonsMath() {
        List<Pair<INDArray,String>> first = NDArrayCreationUtil.getAllTestMatricesWithShape(3, 5, SEED);
        List<Pair<INDArray,String>> firstT = NDArrayCreationUtil.getAllTestMatricesWithShape(5, 3, SEED);
        List<Pair<INDArray,String>> second = NDArrayCreationUtil.getAllTestMatricesWithShape(5, 4, SEED);
        List<Pair<INDArray,String>> secondT = NDArrayCreationUtil.getAllTestMatricesWithShape(4, 5, SEED);
        double[] alpha = {1.0,-0.5,2.5};
        double[] beta = {0.0,-0.25,1.5};
        INDArray cOrig = Nd4j.linspace(1,12,12).reshape(3,4);

        for( int i = 0; i < first.size(); i++ ){
            for( int j = 0; j < second.size(); j++) {
                for( int k = 0; k < alpha.length; k++) {
                    for( int m = 0; m < beta.length; m++) {
                        INDArray cff = Nd4j.create(cOrig.shape(),'f');
                        cff.assign(cOrig);
                        INDArray cft = Nd4j.create(cOrig.shape(),'f');
                        cft.assign(cOrig);
                        INDArray ctf = Nd4j.create(cOrig.shape(),'f');
                        ctf.assign(cOrig);
                        INDArray ctt = Nd4j.create(cOrig.shape(),'f');
                        ctt.assign(cOrig);

                        double a = alpha[k];
                        double b = beta[k];
                        Pair<INDArray, String> p1 = first.get(i);
                        Pair<INDArray, String> p1T = firstT.get(i);
                        Pair<INDArray, String> p2 = second.get(j);
                        Pair<INDArray, String> p2T = secondT.get(j);
                        String errorMsgff = getGemmErrorMsg(i, j, false, false, a,b, p1, p2);
                        String errorMsgft = getGemmErrorMsg(i, j, false, true, a, b, p1, p2T);
                        String errorMsgtf = getGemmErrorMsg(i, j, true, false, a, b, p1T, p2);
                        String errorMsgtt = getGemmErrorMsg(i, j, true, true, a, b, p1T, p2T);
                        System.out.println((String.format("Running iteration %d %d %d %d",i,j,k,m)));
                        assertTrue(errorMsgff, CheckUtil.checkGemm(p1.getFirst(), p2.getFirst(), cff,
                                false, false, a, b, 1e-4, 1e-6));
                        assertTrue(errorMsgft, CheckUtil.checkGemm(p1.getFirst(), p2T.getFirst(), cft,
                                false, true, a, b, 1e-4, 1e-6));
                        assertTrue(errorMsgtf, CheckUtil.checkGemm(p1T.getFirst(), p2.getFirst(), ctf,
                                true, false, a, b, 1e-4, 1e-6));
                        assertTrue(errorMsgtt, CheckUtil.checkGemm(p1T.getFirst(), p2T.getFirst(), ctt,
                                true, true, a, b, 1e-4, 1e-6));
                    }
                }
            }
        }
    }


    private static String getTestWithOpsErrorMsg(int i, int j, String op, Pair<INDArray,String> first, Pair<INDArray,String> second) {
        return i + "," + j + " - " + first.getSecond() + "." + op + "(" + second.getSecond() + ")";
    }

    private static String getGemmErrorMsg(int i, int j, boolean transposeA, boolean transposeB, double alpha, double beta,
                                          Pair<INDArray,String> first, Pair<INDArray,String> second){
        return i + "," + j + " - gemm(tA="+transposeA+",tB="+transposeB+",alpha="+alpha+",beta="+beta+"). A="
                + first.getSecond() + ", B=" + second.getSecond();
    }
}
