/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.autodiff.samediff;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.internal.memory.ArrayCacheMemoryMgr;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.lang.reflect.Field;

import static org.junit.jupiter.api.Assertions.*;

public class MemoryMgrTest extends BaseNd4jTestWithBackends {


    @Override
    public char ordering(){
        return 'c';
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testArrayReuseTooLarge(Nd4jBackend backend) throws Exception {

        ArrayCacheMemoryMgr mmgr = new ArrayCacheMemoryMgr();
        Field f = ArrayCacheMemoryMgr.class.getDeclaredField("maxCacheBytes");
        f.setAccessible(true);
        f.set(mmgr, 1000);

        assertEquals(1000, mmgr.getMaxCacheBytes());

        INDArray[] arrays = new INDArray[100];
        for( int i=0; i<arrays.length; i++ ){
            arrays[i] = Nd4j.create(DataType.FLOAT, 25);        //100 bytes each
        }

        for( int i=0; i<10; i++ ){
            mmgr.release(arrays[i]);
        }

        assertEquals(1000, mmgr.getCurrentCacheSize());
        ArrayCacheMemoryMgr.ArrayStore as = mmgr.getArrayStores().get(DataType.FLOAT);
        assertEquals(1000, as.getBytesSum());
        assertEquals(250, as.getLengthSum());
        assertEquals(10, as.getSize());
        assertEquals(10, mmgr.getLruCache().size());
        assertEquals(10, mmgr.getLruCacheValues().size());


        //At this point: array store is full.
        //If we try to release more, the oldest (first released) values should be closed
        for( int i=0; i<10; i++ ) {
            INDArray toRelease = Nd4j.create(DataType.FLOAT, 25);
            mmgr.release(toRelease);
            //oldest N only should be closed by this point...
            for( int j=0; j<10; j++ ){
                if(j <= i){
                    //Should have been closed
                    assertTrue(arrays[j].wasClosed());
                } else {
                    //Should still be open
                    assertFalse(arrays[j].wasClosed());
                }
            }
        }


        assertEquals(1000, mmgr.getCurrentCacheSize());
        assertEquals(1000, as.getBytesSum());
        assertEquals(250, as.getLengthSum());
        assertEquals(10, as.getSize());
        assertEquals(10, mmgr.getLruCache().size());
        assertEquals(10, mmgr.getLruCacheValues().size());

        //now, allocate some values:
        for( int i = 1; i <= 10; i++) {
            INDArray a1 = mmgr.allocate(true, DataType.FLOAT, 25);
            assertEquals(1000 - i * 100, mmgr.getCurrentCacheSize());
            assertEquals(1000 - i * 100, as.getBytesSum());
            assertEquals(250 - i * 25, as.getLengthSum());
            assertEquals(10 - i, as.getSize());
            assertEquals(10 - i, mmgr.getLruCache().size());
            assertEquals(10 - i, mmgr.getLruCacheValues().size());
        }

        assertEquals(0, mmgr.getCurrentCacheSize());
        assertEquals(0, as.getBytesSum());
        assertEquals(0, as.getLengthSum());
        assertEquals(0, as.getSize());
        assertEquals(0, mmgr.getLruCache().size());
        assertEquals(0, mmgr.getLruCacheValues().size());
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testManyArrays(Nd4jBackend backend){

        ArrayCacheMemoryMgr mmgr = new ArrayCacheMemoryMgr();
        for( int i = 0; i < 1000; i++) {
            mmgr.release(Nd4j.scalar(0));
        }

        assertEquals(4*1000, mmgr.getCurrentCacheSize());
        assertEquals(1000, mmgr.getLruCache().size());
        assertEquals(1000, mmgr.getLruCacheValues().size());

        for( int i = 0; i < 1000; i++ ){
            mmgr.release(Nd4j.scalar(0));
        }

        assertEquals(4*2000, mmgr.getCurrentCacheSize());
        assertEquals(2000, mmgr.getLruCache().size());
        assertEquals(2000, mmgr.getLruCacheValues().size());
    }

}
