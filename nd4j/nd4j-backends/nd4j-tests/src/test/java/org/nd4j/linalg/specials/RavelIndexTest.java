/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.specials;

import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.LongPointer;
import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.nativeblas.NativeOpsHolder;


@Slf4j
@RunWith(Parameterized.class)
public class RavelIndexTest extends BaseNd4jTest {

    DataType initialType;

    public RavelIndexTest(Nd4jBackend backend) {
        super(backend);
        this.initialType = Nd4j.dataType();
    }

    @Before
    public void setUp() {
        Nd4j.setDataType(DataType.FLOAT);
    }

    @After
    public void setDown() {
        Nd4j.setDataType(initialType);
    }

    @Override
    public char ordering() {
        return 'c';
    }


    @Test
    public void ravelIndexesTest() {
        // FIXME: we don't want this test running on cuda for now
        if (Nd4j.getExecutioner().getClass().getCanonicalName().toLowerCase().contains("cuda"))
            return;

        long[]  multiIdxArray = new long[] {
            0,2,7,
            2,36,35,
            3,30,17,
            5,12,22,
            5,43,45,
            6,32,11,
            8,8,32,
            9,29,11,
            5,11,22,
            15,26,16,
            17,48,49,
            24,28,31,
            26,6,23,
            31,21,31,
            35,46,45,
            37,13,14,
            6,38,18,
            7,28,20,
            8,29,39,
            8,32,30,
            9,42,43,
            11,15,18,
            13,18,45,
            29,26,39,
            30,8,25,
            42,31,24,
            28,33,5,
            31,27,1,
            35,43,26,
            36,8,37,
            39,22,14,
            39,24,42,
            42,48,2,
            43,26,48,
            44,23,49,
            45,18,34,
            46,28,5,
            46,32,17,
            48,34,44,
            49,38,39,
        };

        long[] flatIdxArray = new long[] {
                    147,  10955,  14717,  21862,  24055,  27451,  34192,  39841,
                    21792,  64836,  74809, 102791, 109643, 131701, 150265, 156324,
                    27878,  31380,  35669,  35870,  40783,  47268,  55905, 123659,
                    126585, 178594, 119915, 132091, 150036, 151797, 165354, 165522,
                    179762, 182468, 186459, 190294, 195165, 195457, 204024, 208499
        };



        int clipMode = 0;


        long DIM = 3;
        long length = multiIdxArray.length / DIM;
        long[] shape =  new long[] {50, 60, 70};


        DataBuffer multiIdxDB = Nd4j.getDataBufferFactory().createLong(multiIdxArray);
        DataBuffer flatIdxDB = Nd4j.getDataBufferFactory().createLong(flatIdxArray);
        DataBuffer shapeInfo = Nd4j.getShapeInfoProvider().createShapeInformation(shape, DataType.FLOAT).getFirst();

        DataBuffer resultMulti = Nd4j.getDataBufferFactory().createLong(length*DIM);
        DataBuffer resultFlat = Nd4j.getDataBufferFactory().createLong(length);

        NativeOpsHolder.getInstance().getDeviceNativeOps().ravelMultiIndex(null, (LongPointer) multiIdxDB.addressPointer(),
                (LongPointer) resultFlat.addressPointer(), length, (LongPointer) shapeInfo.addressPointer(),clipMode);

        Assert.assertArrayEquals(flatIdxArray, resultFlat.asLong());

        NativeOpsHolder.getInstance().getDeviceNativeOps().unravelIndex(null, (LongPointer) resultMulti.addressPointer(),
                (LongPointer) flatIdxDB.addressPointer(), length, (LongPointer) shapeInfo.addressPointer());

        Assert.assertArrayEquals(multiIdxArray, resultMulti.asLong());


        //testing various clipMode cases
        // clipMode = 0: throw an exception
        try {
            shape[2] = 10;
            shapeInfo = Nd4j.getShapeInfoProvider().createShapeInformation(shape, DataType.FLOAT).getFirst();
            NativeOpsHolder.getInstance().getDeviceNativeOps().ravelMultiIndex(null, (LongPointer) multiIdxDB.addressPointer(),
                    (LongPointer) resultFlat.addressPointer(), length, (LongPointer) shapeInfo.addressPointer(),clipMode);
            Assert.fail("No exception thrown while using CLIP_MODE_THROW.");

        } catch (RuntimeException e) {
            //OK
        }
        // clipMode = 1: wrap around shape
        clipMode = 1;
        multiIdxDB = Nd4j.getDataBufferFactory().createLong(new long[] {3,4, 6,5, 6,9});
        resultFlat = Nd4j.getDataBufferFactory().createLong(3);
        shapeInfo = Nd4j.getShapeInfoProvider().createShapeInformation(new long[] {4, 6}, DataType.FLOAT).getFirst();
        length = 3;

        NativeOpsHolder.getInstance().getDeviceNativeOps().ravelMultiIndex(null, (LongPointer) multiIdxDB.addressPointer(),
                (LongPointer) resultFlat.addressPointer(), length, (LongPointer) shapeInfo.addressPointer(), clipMode);
        Assert.assertArrayEquals(new long[] {22, 17, 15}, resultFlat.asLong());

        // clipMode = 2: clip to shape
        clipMode = 2;
        multiIdxDB = Nd4j.getDataBufferFactory().createLong(new long[] {3,4, 6,5, 6,9});
        resultFlat = Nd4j.getDataBufferFactory().createLong(3);
        shapeInfo = Nd4j.getShapeInfoProvider().createShapeInformation(new long[] {4, 6}, DataType.FLOAT).getFirst();
        length = 3;

        NativeOpsHolder.getInstance().getDeviceNativeOps().ravelMultiIndex(null, (LongPointer) multiIdxDB.addressPointer(),
                (LongPointer) resultFlat.addressPointer(), length, (LongPointer) shapeInfo.addressPointer(), clipMode);

        Assert.assertArrayEquals(new long[] {22, 23, 23}, resultFlat.asLong());



    }

}
