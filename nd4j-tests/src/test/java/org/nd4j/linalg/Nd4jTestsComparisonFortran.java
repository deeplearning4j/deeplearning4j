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
import org.nd4j.linalg.checkutil.CheckUtil;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

/**Tests comparing Nd4j ops to other libraries
 */
public  class Nd4jTestsComparisonFortran extends BaseNd4jTest {
    private static Logger log = LoggerFactory.getLogger(Nd4jTestsComparisonFortran.class);

    public static final int SEED = 123;

    public Nd4jTestsComparisonFortran() {
    }

    public Nd4jTestsComparisonFortran(String name) {
        super(name);
    }

    public Nd4jTestsComparisonFortran(Nd4jBackend backend) {
        super(backend);
    }

    public Nd4jTestsComparisonFortran(String name, Nd4jBackend backend) {
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
        return 'f';
    }
    
    @Test
    public void testMmulWithOpsCommonsMath(){
    	List<Pair<INDArray,String>> first = CheckUtil.getAllTestMatricesWithShape(3, 5, SEED);
    	List<Pair<INDArray,String>> second = CheckUtil.getAllTestMatricesWithShape(5, 4, SEED);
    	for( int i=0; i<first.size(); i++ ){
    		for( int j=0; j<second.size(); j++ ){
    			Pair<INDArray,String> p1 = first.get(i);
    			Pair<INDArray,String> p2 = second.get(i);
    			String errorMsg = getTestWithOpsErrorMsg(i,j,"mmul",p1,p2);
    			assertTrue(errorMsg, CheckUtil.checkMmul(p1.getFirst(), p2.getFirst(), 1e-4, 1e-6));
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
    			Pair<INDArray,String> p2 = second.get(i);
    			String errorMsg1 = getTestWithOpsErrorMsg(i,j,"add",p1,p2);
    			String errorMsg2 = getTestWithOpsErrorMsg(i,j,"sub",p1,p2);
    			assertTrue(errorMsg1,CheckUtil.checkAdd(p1.getFirst(), p2.getFirst(), 1e-4, 1e-6));
                assertTrue(errorMsg2,CheckUtil.checkSubtract(p1.getFirst(), p2.getFirst(), 1e-4, 1e-6));
    		}
    	}
    }

    private static String getTestWithOpsErrorMsg(int i, int j, String op, Pair<INDArray,String> first, Pair<INDArray,String> second) {
        return i + "," + j + " - " + first.getSecond() + "." + op + "(" + second.getSecond() + ")";
    }
}
