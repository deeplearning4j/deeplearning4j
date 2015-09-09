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


import org.apache.commons.math3.util.Pair;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.checkutil.CheckUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

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
    public void testMmulWithOpsCommonsMath(){
        List<Pair<INDArray,String>> first = CheckUtil.getAllTestMatricesWithShape(3, 5, SEED);
        List<Pair<INDArray,String>> second = CheckUtil.getAllTestMatricesWithShape(5, 4, SEED);
        for( int i = 0; i < first.size(); i++ ){
            for( int j = 0; j < second.size(); j++ ){
                Pair<INDArray,String> p1 = first.get(i);
                Pair<INDArray,String> p2 = second.get(j);
                String errorMsg = getTestWithOpsErrorMsg(i,j,"mmul",p1,p2);
                assertTrue(errorMsg, CheckUtil.checkMmul(p1.getFirst(), p2.getFirst(), 1e-4, 1e-6));
            }
        }
    }
    
    @Test
    public void testGemmWithOpsCommonsMath(){
    	List<Pair<INDArray,String>> first = CheckUtil.getAllTestMatricesWithShape(3, 5, SEED);
    	List<Pair<INDArray,String>> second = CheckUtil.getAllTestMatricesWithShape(5, 4, SEED);
        double[] alpha = {-0.5,1.0,2.5};
        double[] beta = {0.0,1.0,3.5};
        INDArray cOrig = Nd4j.create(new int[]{3,4});
        //TODO random values
    	for( int i = 0; i < first.size(); i++ ){
    		for( int j = 0; j < second.size(); j++ ){
                for( int k=0; k<alpha.length; k++ ) {
                    for( int m=0; m<beta.length; m++ ) {
                        INDArray cff = Nd4j.create(cOrig.shape(),'f');
                        cff.assign(cOrig);
                        INDArray cft = Nd4j.create(cOrig.shape(),'f');
                        cff.assign(cOrig);
                        INDArray ctf = Nd4j.create(cOrig.shape(),'f');
                        cff.assign(cOrig);
                        INDArray ctt = Nd4j.create(cOrig.shape(),'f');
                        cff.assign(cOrig);

                        double a = alpha[k];
                        double b = beta[k];
                        Pair<INDArray, String> p1 = first.get(i);
                        Pair<INDArray, String> p2 = second.get(j);
                        String errorMsgff = getGemmErrorMsg(i, j, false, false, a,b, p1, p2);
                        String errorMsgft = getGemmErrorMsg(i, j, false, true, a, b, p1, p2);
                        String errorMsgtf = getGemmErrorMsg(i, j, true, false, a, b, p1, p2);
                        String errorMsgtt = getGemmErrorMsg(i, j, true, true, a, b, p1, p2);

                        assertTrue(errorMsgff, CheckUtil.checkGemm(p1.getFirst(), p2.getFirst(), cff,
                                false, false, a, b, 1e-4, 1e-6));
                        assertTrue(errorMsgft, CheckUtil.checkGemm(p1.getFirst(), p2.getFirst(), cft,
                                false, true, a, b, 1e-4, 1e-6));
                        assertTrue(errorMsgtf, CheckUtil.checkGemm(p1.getFirst(), p2.getFirst(), ctf,
                                true, false, a, b, 1e-4, 1e-6));
                        assertTrue(errorMsgtt, CheckUtil.checkGemm(p1.getFirst(), p2.getFirst(), ctt,
                                true, true, a, b, 1e-4, 1e-6));
                    }
                }
    		}
    	}
    }
    
    @Test
    public void testAddSubtractWithOpsCommonsMath() {
    	List<Pair<INDArray,String>> first = CheckUtil.getAllTestMatricesWithShape(3, 5, SEED);
    	List<Pair<INDArray,String>> second = CheckUtil.getAllTestMatricesWithShape(3, 5, SEED);
    	for( int i=0; i<first.size(); i++ ){
    		for( int j=0; j<second.size(); j++ ){
    			Pair<INDArray,String> p1 = first.get(i);
    			Pair<INDArray,String> p2 = second.get(j);
    			String errorMsg1 = getTestWithOpsErrorMsg(i, j, "add", p1, p2);
    			String errorMsg2 = getTestWithOpsErrorMsg(i,j,"sub",p1,p2);
    			assertTrue(errorMsg1,CheckUtil.checkAdd(p1.getFirst(), p2.getFirst(), 1e-4, 1e-6));
                assertTrue(errorMsg2,CheckUtil.checkSubtract(p1.getFirst(), p2.getFirst(), 1e-4, 1e-6));
    		}
    	}
    }

    @Test
    public void testMulDivOnCheckUtilMatrices(){
        List<Pair<INDArray,String>> first = CheckUtil.getAllTestMatricesWithShape(3, 5, SEED);
        List<Pair<INDArray,String>> second = CheckUtil.getAllTestMatricesWithShape(3, 5, SEED);
        for( int i=0; i<first.size(); i++ ){
            for( int j=0; j<second.size(); j++ ){
                Pair<INDArray,String> p1 = first.get(i);
                Pair<INDArray,String> p2 = second.get(j);
                String errorMsg1 = getTestWithOpsErrorMsg(i,j,"mul",p1,p2);
                String errorMsg2 = getTestWithOpsErrorMsg(i,j,"div",p1,p2);
                assertTrue(errorMsg1,CheckUtil.checkMulManually(p1.getFirst(), p2.getFirst(), 1e-4, 1e-6));
                assertTrue(errorMsg2,CheckUtil.checkDivManually(p1.getFirst(), p2.getFirst(), 1e-4, 1e-6));
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
