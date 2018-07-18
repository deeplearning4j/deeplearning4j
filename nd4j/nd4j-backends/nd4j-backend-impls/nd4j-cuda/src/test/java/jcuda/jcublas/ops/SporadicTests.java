/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package jcuda.jcublas.ops;

import lombok.extern.slf4j.Slf4j;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.jita.allocator.impl.AllocationPoint;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.GridExecutioner;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastSubOp;
import org.nd4j.linalg.api.ops.impl.indexaccum.IAMax;
import org.nd4j.linalg.api.ops.impl.scalar.ScalarAdd;
import org.nd4j.linalg.api.ops.impl.transforms.IsMax;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.jcublas.ops.executioner.CudaGridExecutioner;
import org.nd4j.linalg.util.DeviceLocalNDArray;

import java.io.File;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.*;
import static org.nd4j.linalg.api.shape.Shape.newShapeNoCopy;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class SporadicTests {

    @Before
    public void setUp() throws Exception {
        //CudaEnvironment.getInstance().getConfiguration().enableDebug(true).setVerbose(false);
    }

    @Test
    public void testIsMax1() throws Exception {
        int[] shape = new int[]{2,2};
        int length = 4;
        int alongDimension = 0;

        INDArray arrC = Nd4j.linspace(1,length, length).reshape('c',shape);
        Nd4j.getExecutioner().execAndReturn(new IsMax(arrC, alongDimension));

        //System.out.print(arrC);
        assertEquals(0.0, arrC.getDouble(0), 0.1);
        assertEquals(0.0, arrC.getDouble(1), 0.1);
        assertEquals(1.0, arrC.getDouble(2), 0.1);
        assertEquals(1.0, arrC.getDouble(3), 0.1);
    }

    @Test
    public void randomStrangeTest() {
        DataBuffer.Type type = Nd4j.dataType();
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE);

        int a=9;
        int b=2;
        int[] shapes = new int[a];
        for (int i = 0; i < a; i++) {
            shapes[i] = b;
        }
        INDArray c = Nd4j.linspace(1, (int) (100 * 1 + 1 + 2), (int) Math.pow(b, a)).reshape(shapes);
        c=c.sum(0);
        double[] d = c.data().asDouble();
        System.out.println("d: " + Arrays.toString(d));

        DataTypeUtil.setDTypeForContext(type);
    }

    @Test
    public void testBroadcastWithPermute(){
        Nd4j.getRandom().setSeed(12345);
        int length = 4*4*5*2;
        INDArray arr = Nd4j.linspace(1,length,length).reshape('c',4,4,5,2).permute(2,3,1,0);
//        INDArray arr = Nd4j.linspace(1,length,length).reshape('f',4,4,5,2).permute(2,3,1,0);
        Nd4j.getExecutioner().commit();
        INDArray arrDup = arr.dup('c');
        Nd4j.getExecutioner().commit();

        INDArray row = Nd4j.rand(1,2);
        assertEquals(row.length(), arr.size(1));
        assertEquals(row.length(), arrDup.size(1));

        assertEquals(arr,arrDup);



        INDArray first =  Nd4j.getExecutioner().execAndReturn(new BroadcastSubOp(arr,    row, Nd4j.createUninitialized(arr.shape(), 'c'), 1));
        INDArray second = Nd4j.getExecutioner().execAndReturn(new BroadcastSubOp(arrDup, row, Nd4j.createUninitialized(arr.shape(), 'c'), 1));

        System.out.println("A1: " + Arrays.toString(arr.shapeInfoDataBuffer().asInt()));
        System.out.println("A2: " + Arrays.toString(first.shapeInfoDataBuffer().asInt()));
        System.out.println("B1: " + Arrays.toString(arrDup.shapeInfoDataBuffer().asInt()));
        System.out.println("B2: " + Arrays.toString(second.shapeInfoDataBuffer().asInt()));

        INDArray resultSameStrides = Nd4j.zeros(new int[]{4,4,5,2},'c').permute(2,3,1,0);
        assertArrayEquals(arr.stride(), resultSameStrides.stride());
        INDArray third = Nd4j.getExecutioner().execAndReturn(new BroadcastSubOp(arr, row, resultSameStrides, 1));

        assertEquals(second, third);    //Original and result w/ same strides: passes
        assertEquals(first,second);     //Original and result w/ different strides: fails
    }

    @Test
    public void testBroadcastEquality1() {
        INDArray array = Nd4j.zeros(new int[]{4, 5}, 'f');
        INDArray array2 = Nd4j.zeros(new int[]{4, 5}, 'f');
        INDArray row = Nd4j.create(new float[]{1, 2, 3, 4, 5});

        array.addiRowVector(row);

        System.out.println(array);

        System.out.println("-------");

        ScalarAdd add = new ScalarAdd(array2, row, array2, array2.length(), 0.0f);
        add.setDimension(0);
        Nd4j.getExecutioner().exec(add);

        System.out.println(array2);
        assertEquals(array, array2);
    }

    @Test
    public void testBroadcastEquality2() {
        INDArray array = Nd4j.zeros(new int[]{4, 5}, 'c');
        INDArray array2 = Nd4j.zeros(new int[]{4, 5}, 'c');
        INDArray column = Nd4j.create(new float[]{1, 2, 3, 4}).reshape(4,1);

        array.addiColumnVector(column);

        System.out.println(array);

        System.out.println("-------");

        ScalarAdd add = new ScalarAdd(array2, column, array2, array2.length(), 0.0f);
        add.setDimension(1);
        Nd4j.getExecutioner().exec(add);

        System.out.println(array2);
        assertEquals(array, array2);

    }

    @Test
    public void testIAMax1() throws Exception {
        INDArray arrayX = Nd4j.rand('c', 128000, 4);

        Nd4j.getExecutioner().exec(new IAMax(arrayX), 1);

        long time1 = System.nanoTime();
        for (int i = 0; i < 10000; i++) {
            Nd4j.getExecutioner().exec(new IAMax(arrayX), 1);
        }
        long time2 = System.nanoTime();

        System.out.println("Time: " + ((time2 - time1) / 10000));
    }

    @Test
    public void testLocality() {
        INDArray array = Nd4j.create(new float[]{1,2,3,4,5,6,7,8,9});

        AllocationPoint point = AtomicAllocator.getInstance().getAllocationPoint(array);
        assertEquals(true, point.isActualOnDeviceSide());

        INDArray arrayR = array.reshape('f', 3, 3);

        AllocationPoint pointR = AtomicAllocator.getInstance().getAllocationPoint(arrayR);
        assertEquals(true, pointR.isActualOnDeviceSide());

        INDArray arrayS = Shape.newShapeNoCopy(array,new int[]{3,3}, true);

        AllocationPoint pointS = AtomicAllocator.getInstance().getAllocationPoint(arrayS);
        assertEquals(true, pointS.isActualOnDeviceSide());

        INDArray arrayL = Nd4j.create(new int[]{3,4,4,4},'c');

        AllocationPoint pointL = AtomicAllocator.getInstance().getAllocationPoint(arrayL);
        assertEquals(true, pointL.isActualOnDeviceSide());
    }

    @Test
    public void testEnvironment() throws Exception {
        INDArray array = Nd4j.zeros(new int[]{4, 5}, 'f');
        Properties properties = Nd4j.getExecutioner().getEnvironmentInformation();

        System.out.println("Props: " + properties.toString());
    }


    /**
     * This is special test that checks for memory alignment
     * @throws Exception
     */
    @Test
    @Ignore
    public void testDTypeSpam() throws Exception {
        Random rnd = new Random();
        for(int i = 0; i < 100; i++) {
            DataTypeUtil.setDTypeForContext(DataBuffer.Type.FLOAT);
            float rand[] = new float[rnd.nextInt(10) + 1];
            for (int x = 0; x < rand.length; x++) {
                rand[x] = rnd.nextFloat();
            }
            Nd4j.getConstantHandler().getConstantBuffer(rand);

            int shape[] = new int[rnd.nextInt(3)+2];
            for (int x = 0; x < shape.length; x++) {
                shape[x] = rnd.nextInt(100) + 2;
            }

            DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE);
            INDArray array = Nd4j.rand(shape);
            BooleanIndexing.applyWhere(array, Conditions.lessThan(rnd.nextDouble()), rnd.nextDouble());
        }
    }

    @Test
    public void testIsView() {
        INDArray array = Nd4j.zeros(100, 100);

        assertFalse(array.isView());
    }


    @Test
    public void testReplicate1() throws Exception {
        INDArray array = Nd4j.create(new float[]{1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f});
        INDArray exp = Nd4j.create(new float[]{2f, 2f, 2f, 2f, 2f, 2f, 2f, 2f, 2f, 2f, 2f, 2f, 2f, 2f, 2f, 2f, 2f, 2f, 2f, 2f});

        log.error("Array length: {}", array.length());

        int numDevices = Nd4j.getAffinityManager().getNumberOfDevices();

        final DeviceLocalNDArray locals = new DeviceLocalNDArray(array);

        Thread[] threads = new Thread[numDevices];
        for (int t = 0; t < numDevices; t++) {
            threads[t] = new Thread(new Runnable() {
                @Override
                public void run() {
                    locals.get().addi(1f);
                    locals.get().addi(0f);
                }
            });
            threads[t].start();
        }


        for (int t = 0; t < numDevices; t++) {
            threads[t].join();
        }


        for (int t = 0; t < numDevices; t++) {
            exp.addi(0.0f);
            assertEquals(exp, locals.get(t));
        }
    }

    @Test
    public void testReplicate2() throws Exception {
        DataBuffer buffer = Nd4j.createBuffer(new float[] {1f, 1f, 1f, 1f, 1f});

        DataBuffer buffer2 = Nd4j.getAffinityManager().replicateToDevice(1, buffer);

        assertEquals(1f, buffer2.getFloat(0), 0.001f);
    }


    @Test
    public void testReplicate3() throws Exception {
        INDArray array = Nd4j.ones(10, 10);
        INDArray exp = Nd4j.create(10).assign(10f);

        log.error("Array length: {}", array.length());

        int numDevices = Nd4j.getAffinityManager().getNumberOfDevices();

        final DeviceLocalNDArray locals = new DeviceLocalNDArray(array);

        Thread[] threads = new Thread[numDevices];
        for (int t = 0; t < numDevices; t++) {
            threads[t] = new Thread(new Runnable() {
                @Override
                public void run() {

                    AllocationPoint point = AtomicAllocator.getInstance().getAllocationPoint(locals.get());
                    log.error("Point deviceId: {}; current deviceId: {}", point.getDeviceId(), Nd4j.getAffinityManager().getDeviceForCurrentThread());


                    INDArray sum = locals.get().sum(1);
                    INDArray localExp = Nd4j.create(10).assign(10f);

                    assertEquals(localExp, sum);
                }
            });
            threads[t].start();
        }


        for (int t = 0; t < numDevices; t++) {
            threads[t].join();
        }


        for (int t = 0; t < numDevices; t++) {

            AllocationPoint point = AtomicAllocator.getInstance().getAllocationPoint(locals.get(t));
            log.error("Point deviceId: {}; current deviceId: {}", point.getDeviceId(), Nd4j.getAffinityManager().getDeviceForCurrentThread());

            exp.addi(0.0f);
            assertEquals(exp, locals.get(t).sum(0));

            log.error("Point after: {}", point.getDeviceId());
        }
    }


    @Test
    public void testReplicate4() throws Exception {
        INDArray array = Nd4j.create(3,3);

        array.getRow(1).putScalar(0, 1f);
        array.getRow(1).putScalar(1, 1f);
        array.getRow(1).putScalar(2, 1f);

        final DeviceLocalNDArray locals = new DeviceLocalNDArray(array);

        int numDevices = Nd4j.getAffinityManager().getNumberOfDevices();
        for (int t = 0; t < numDevices; t++) {
            assertEquals(3, locals.get(t).sumNumber().floatValue(), 0.001f);
        }
    }


    @Test
    public void testReplicate5() throws Exception {
        INDArray array = Nd4j.create(3, 3);

        log.error("Original: Host pt: {}; Dev pt: {}", AtomicAllocator.getInstance().getAllocationPoint(array).getPointers().getHostPointer().address(), AtomicAllocator.getInstance().getAllocationPoint(array).getPointers().getDevicePointer().address());

        final DeviceLocalNDArray locals = new DeviceLocalNDArray(array);



        int numDevices = Nd4j.getAffinityManager().getNumberOfDevices();
        for (int t = 0; t < numDevices; t++) {
            log.error("deviceId: {}; Host pt: {}; Dev pt: {}", t, AtomicAllocator.getInstance().getAllocationPoint(locals.get(t)).getPointers().getHostPointer().address(), AtomicAllocator.getInstance().getAllocationPoint(locals.get(t)).getPointers().getDevicePointer().address());
        }


        Thread[] threads = new Thread[numDevices];
        for (int t = 0; t < numDevices; t++) {
            threads[t] = new Thread(new Runnable() {
                @Override
                public void run() {
                    AllocationPoint point = AtomicAllocator.getInstance().getAllocationPoint(locals.get());
                    log.error("deviceId: {}; Host pt: {}; Dev pt: {}", Nd4j.getAffinityManager().getDeviceForCurrentThread(), point.getPointers().getHostPointer().address(), point.getPointers().getDevicePointer().address());

                }
            });
            threads[t].start();
        }


        for (int t = 0; t < numDevices; t++) {
            threads[t].join();
        }
    }


    @Test
    public void testEnvInfo() throws Exception {
        Properties props = Nd4j.getExecutioner().getEnvironmentInformation();

        List<Map<String, Object>> list = (List<Map<String,Object>>) props.get("cuda.devicesInformation");
        for (Map<String, Object> map: list) {
            log.error("devName: {}", map.get("cuda.deviceName"));
            log.error("totalMem: {}", map.get("cuda.totalMemory"));
            log.error("freeMem: {}", map.get("cuda.freeMemory"));
            System.out.println();
        }
    }

    @Test
    public void testStd() {
        INDArray values = Nd4j.linspace(1, 4, 4).transpose();

        double corrected = values.std(true, 0).getDouble(0);
        double notCorrected = values.std(false, 0).getDouble(0);

        System.out.println(String.format("Corrected: %f, non corrected: %f", corrected, notCorrected));

    }

    @Ignore
    @Test
    public void testHalf19() {
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.HALF);
        INDArray first = Nd4j.rand(20, 10);
        INDArray second = Nd4j.rand(3, 20);
        DataSet data = new DataSet(first, second);

        data.normalize();
    }


    @Test
    public void testDebugEdgeCase(){
        INDArray l1 = Nd4j.create(new double[]{-0.2585039112684677,-0.005179485353710878,0.4348343401770497,0.020356532375728764,-0.1970793298488186});
        INDArray l2 = Nd4j.create(3,l1.size(1));

        INDArray p1 = Nd4j.create(new double[]{1.3979850406519119,0.6169451410155852,1.128993957530918,0.21000426084450596,0.3171215178932696});
        INDArray p2 = Nd4j.create(3, p1.size(1));

        for( int i=0; i<3; i++ ){
            l2.putRow(i, l1);
            p2.putRow(i, p1);
        }

        INDArray s1 = scoreArray(l1, p1);
        INDArray s2 = scoreArray(l2, p2);

        //Outputs here should be identical:
        System.out.println(Arrays.toString(s1.data().asDouble()));
        System.out.println(Arrays.toString(s2.getRow(0).dup().data().asDouble()));
    }

    public static INDArray scoreArray(INDArray labels, INDArray preOutput) {
        INDArray yhatmag = preOutput.norm2(1);

        INDArray scoreArr = preOutput.mul(labels);
        scoreArr.diviColumnVector(yhatmag);

        return scoreArr;
    }

    @Test
    public void testDebugEdgeCase2(){
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE);
        INDArray l1 = Nd4j.create(new double[]{-0.2585039112684677,-0.005179485353710878,0.4348343401770497,0.020356532375728764,-0.1970793298488186});
        INDArray l2 = Nd4j.create(2,l1.size(1));

        INDArray p1 = Nd4j.create(new double[]{1.3979850406519119,0.6169451410155852,1.128993957530918,0.21000426084450596,0.3171215178932696});
        INDArray p2 = Nd4j.create(2, p1.size(1));

        for( int i=0; i<2; i++ ){
            l2.putRow(i, l1);
            p2.putRow(i, p1);
        }

        INDArray norm2_1 = l1.norm2(1);
        System.out.println("Queue: " + ((CudaGridExecutioner) Nd4j.getExecutioner()).getQueueLength());

        INDArray temp1 = p1.mul(l1);

        System.out.println("Queue: " + ((CudaGridExecutioner) Nd4j.getExecutioner()).getQueueLength());

//        if (Nd4j.getExecutioner() instanceof CudaGridExecutioner)
//            ((CudaGridExecutioner) Nd4j.getExecutioner()).flushQueueBlocking();

        INDArray out1 = temp1.diviColumnVector(norm2_1);
        System.out.println("------");

        Nd4j.getExecutioner().commit();

        INDArray norm2_2 = l2.norm2(1);

        System.out.println("norm2_1: " + Arrays.toString(norm2_1.data().asDouble()));
        System.out.println("norm2_2: " + Arrays.toString(norm2_2.data().asDouble()));

        INDArray temp2 = p2.mul(l2);



        System.out.println("temp1: " + Arrays.toString(temp1.data().asDouble()));
        System.out.println("temp2: " + Arrays.toString(temp2.data().asDouble()));

        INDArray out2 = temp2.diviColumnVector(norm2_2);

        //Outputs here should be identical:
        System.out.println(Arrays.toString(out1.data().asDouble()));
        System.out.println(Arrays.toString(out2.getRow(0).dup().data().asDouble()));
    }

    @Test
    public void testSum(){
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE);
        System.out.println("dType: "+ Nd4j.dataType());
        double[] d = new double[]{0.0018527202080054288, 4.860460399681257E-5, 1.0866740014537973E-4, -0.0067027589698796745, 3.137875745180366E-4, -0.004068275565124544, -0.01787441478759584, 0.008485829165871582, -2.763756128155635E-4, 2.871786662038523E-4, -0.010483973019250817, 0.007756981321203987, -0.0017533316296846166, -0.013154552997235138, 0.002664089383318023, 0.003219604745689706, -0.017140063751978196, -0.0402780728371146, -0.024062552380901596, 0.0034055167376910362, -0.0014209322438402164, -0.0019807697611663373, -2.2838830354674253E-4, 0.00802614947703168, 6.809417793628905E-5, -3.5682443741929696E-4, 2.3642116161687557E-4, 0.0020248602841291333, 0.008922488929012673, -1.8730312287067906E-6, 2.6969347916614046E-5, -1.3560804909521155E-5, -0.0019803959188298536, -0.011388435316124648, 0.009815024112605878, -6.217212819848868E-5, 7.198174495385047E-6, 7.859570666088778E-4, 3.2352438373925256E-4, 0.009310926419061586, 0.001285919484459703, -0.004614530932162568, -5.693929364499898E-4, 4.436935763832914E-5, 0.010423203318809186, 0.006593852752045009, 0.0063445124848706584, 9.737683314182195E-5, -7.002675823907349E-4, 0.0010906784650723032, -9.972152373258224E-6, 0.00871521334612937, 0.0015878927877041975, 3.5864863535235535E-4, -4.398790476749721E-5, -7.77853455185052E-5, 1.8862217750434992E-4, 0.0224868440061588, 0.0073318858545188175, -8.220926236861101E-4, -2.4336360596325374E-4, -0.0018348955616861627, 0.011423225743787646, -0.0016207645948113378, 2.289915435117371E-4, -7.122130486259979E-4, -4.94058936059287E-4, 0.004245767850547438, 2.1598406094246788E-4, 0.0014093429117757112, 3.1948093888499473E-4, -1.327894312872927E-4, 5.756401064075624E-4, -0.013501868757425933, 0.08022280647460137, -0.025763510735921924, 0.2147635435756625, -3.570893204705811E-4, 0.23699343725699218, 0.02005726530793397, 0.2233494849035487, 0.0015628679046820334, 0.03686828571588657, -0.034884254322163376, -0.04580585504492872, 0.022492109246861913, 0.6122906576027609, 0.0013512843074173794, 0.009833469844281123, -0.12754922577196826, -0.05866094108326281, 0.00786015783509335, 0.012943402024682067, -0.04138337949224019, 0.16422596234194609, -0.003224047448184361, -0.013553967826667544, 0.024567523776697443, 0.003119569763001505, 0.06676632404231841, 0.0019518418481909879, -3.546570152995131E-4, 8.843184961729061E-4, 0.02791605441470666, -0.013688049930718094, 0.03237370158354087, 7.749693316768275E-4, -0.006175397798237791, -0.001425650837729542, -7.358356122518933E-4, -5.924546696292049E-4, 2.572174974203492E-4, 0.008635399542952251, -0.00785894020636433, 0.00611004654858908, 3.849937280461072E-4, -0.0011280073492511923, -0.014039863611056342, 0.005258910284221449, -0.0012716079353840685, -0.005880609969998075, 0.03884904612026859, -0.007808162270479559, 0.13764734350512128, -0.0955607452917015, 0.01739042887598923, 0.003176716700283583, -8.845189001196553E-5, 0.059890266991132, -0.011719738031782573, -0.009720651901132008, -0.020271048497565218, 3.5861474460486776E-5, 0.003234054136867597, -0.016855942686723118, -0.04109181803561225, 0.03929335910336556, -0.002045944958743484, 4.986319734224706E-6, -2.0719501403766647E-5, 0.022377318509545937, -0.007592601387358396, 4.490315393644052E-4, 9.033852118576955E-5, 4.621091068084668E-6, -5.247702006473915E-4, -7.902654461500924E-4, -0.0011914084606579713, 0.0030085580689989877, -4.246971856810759E-4, -1.2340215512440867E-4, 0.0019671074593817285, -6.010387216740781E-4, 0.013650305790487045, -0.0011454153127967719, -0.007189180788631945, 2.870289907492301E-6, 4.3693088414999864E-4, 0.01200434591332941, -0.014509596674846678, -0.0029357117029629866, -2.1150207332328822E-4, -0.00315536512642124, 1.0374814880225154E-4, -0.0034757406691398496, 0.011599985159323294, 1.2969970680596453E-4, 9.964327556021442E-4, -0.001849649601501932, 0.002689358375591656, 0.012896200751328621, 0.007476029001401352, -0.0033194177760658377, -4.3432827454975976E-4, 3.411369387610943E-4, -4.103832908635317E-4, 0.007055642948203781, -0.0015501810107658967, -0.005752034090813254, -2.844831713420882E-4, -9.563438979460705E-5, -0.02284555356663203, -0.009025504086580169, -0.1559083024105329, 0.12294355422935457, -8.708345100849238E-4, 0.02784682111718311, 0.09887344727692746, 0.1984110780215329, -0.0019539047730033083, 0.436534119185953, -0.0022943880212978763, 0.0033303334626212217, -0.47305986663738375, 0.2870128297740214, -0.4852364244335913, -0.1966639932906117, -0.011543131716351632, -0.037961570290375855, 0.7991053621370379, -0.0965493466734368, 0.14022527291688097, -0.15353621266599798, 0.032127740955076554, -0.03391229079838272, 0.04220928870735664, -0.10022115665949234, -0.0060843857983522015, 0.05969884290137722, -0.001513774894231756, 0.003573617155056928, -0.030126515163639428, 0.006604847374388239, -0.01685524264155275, -0.015135550991685925, -0.002122525156000015};
        int[] shape = {2, 108};
        int[] stride = {108, 1};
        char order = 'c';
        INDArray arr = Nd4j.create(d, shape, stride, 0, order );

        double[] exp = new double[2];
        for( int i=0; i<shape[1]; i++ ){
            exp[0] += arr.getDouble(0,i);
            exp[1] += arr.getDouble(1,i);
        }

        System.out.println("Expected: " + Arrays.toString(exp));
        System.out.println("Actual:   " + Arrays.toString(arr.sum(1).data().asDouble()));
    }


    @Test
    public void testDataSetSaveLost() throws Exception {
        INDArray features = Nd4j.linspace(1, 16 * 784, 16 * 784).reshape(16, 784);
        INDArray labels = Nd4j.linspace(1, 160, 160).reshape(16, 10);

        for (int i = 0; i < 100; i++) {
            DataSet ds = new DataSet(features, labels);

            File tempFile = File.createTempFile("dataset", "temp");
            tempFile.deleteOnExit();

            ds.save(tempFile);

            DataSet restore = new DataSet();
            restore.load(tempFile);

            assertEquals(features, restore.getFeatureMatrix());
            assertEquals(labels, restore.getLabels());

        }
    }

    @Test
    public void testEps() throws Exception {

        DataTypeUtil.setDTypeForContext(DataBuffer.Type.HALF);

        INDArray arr = Nd4j.create(new double[]{0,0,0,1,1,1,2,2,2});

        System.out.println(arr.eps(0.0));
        System.out.println(arr.eps(1.0));
        System.out.println(arr.eps(2.0));
    }

    @Test
    public void testNeg() {
        INDArray rnd = Nd4j.rand(2, 2);
        System.out.println(rnd.equals(rnd.neq(1)));
    }

    @Test
    public void testEq() {
        INDArray z = Nd4j.ones(2, 2)
                .eq(2);

        Nd4j.getExecutioner().commit();

        System.out.println("Z: " + z);
    }

    @Test
    public void testCrash() throws Exception {
        System.out.println("Executor: " + Nd4j.getExecutioner().getClass().getSimpleName());
        int shape[] = new int[]{1, 3, 150, 150};
        INDArray img = Nd4j.create(shape);
        INDArray lbl = Nd4j.create(205);
        AtomicInteger cnt = new AtomicInteger(0);

        while (cnt.get() < 16) {
            System.out.println("Iteration: " + cnt.getAndIncrement());
            getBatch(img, lbl, 128);
        }
    }

    @Test
    public void testAffinityManager() {
        Nd4j.getMemoryManager().setAutoGcWindow(127);

        assertEquals(127, CudaEnvironment.getInstance().getConfiguration().getNoGcWindowMs());
    }

    @Test
    public void testPrintOut() throws Exception {
        Nd4j.create(100);

        Nd4j.getExecutioner().printEnvironmentInformation();

        log.info("-------------------------------------");
        Nd4j.create(500);

        Nd4j.getExecutioner().printEnvironmentInformation();
    }

    @Test
    public void testReduceX() throws Exception {
        CudaEnvironment.getInstance().getConfiguration().setMaximumGridSize(11);
        INDArray x = Nd4j.create(500, 500);
        INDArray exp_0 = Nd4j.linspace(1, 500, 500);
        INDArray exp_1 = Nd4j.create(500).assign(250.5);

        x.addiRowVector(Nd4j.linspace(1, 500, 500));

        assertEquals(exp_0, x.mean(0));
        assertEquals(exp_1, x.mean(1));

        assertEquals(250.5, x.meanNumber().doubleValue(), 1e-5);
    }

    @Test
    public void testIndexReduceX() throws Exception {
        CudaEnvironment.getInstance().getConfiguration().setMaximumGridSize(11);
        INDArray x = Nd4j.create(500, 500);
        INDArray exp_0 = Nd4j.create(500).assign(0);
        INDArray exp_1 = Nd4j.create(500).assign(499);

        x.addiRowVector(Nd4j.linspace(1, 500, 500));

        assertEquals(exp_0, Nd4j.argMax(x, 0));
        assertEquals(exp_1, Nd4j.argMax(x, 1));
    }

    @Test
    public void testInf() {
        INDArray x = Nd4j.create(10).assign(0.0);

        x.muli(0.0);

        log.error("X: {}", x);
    }

    @Test
    public void testTreo1() {
        INDArray points = Nd4j.rand(100000, 300);
        INDArray q = Nd4j.rand(10000, 300);

        System.out.println("----------------");
        ArrayList<Float> floats1 = new ArrayList<>();
        List<Long> results = new ArrayList<>();
        for (int i = 0; i < 10000; i++) {
            long time1 = System.currentTimeMillis();
            INDArray gemm = points.mmul(q.getRow(i).transpose());
            float[] floats = gemm.data().asFloat();
            long time2 = System.currentTimeMillis();
            /*for (int k = 0; k < floats.length; k++) {
                floats1.add(floats[k]);
            }

            floats1.clear();*/
            results.add(time2 - time1);
        }

        log.error("p50: {}", results.get(results.size() / 2));
    }

    public DataSet getBatch(INDArray input, INDArray label, int batchSize) {
        List<INDArray> inp = new ArrayList<>();
        List<INDArray> lab = new ArrayList<>();
        for (int i = 0; i < batchSize; i++) {
            inp.add(input);
            lab.add(label);
        }

        DataSet ds = getTransformation(inp, inp);
        return ds;
    }

    public DataSet getTransformation(List<INDArray> inp , List<INDArray> lab){
        DataSet ret =  new DataSet(Nd4j.vstack(inp.toArray(new INDArray[0])), Nd4j.vstack(lab.toArray(new INDArray[0])));
        return ret;
    }
}
