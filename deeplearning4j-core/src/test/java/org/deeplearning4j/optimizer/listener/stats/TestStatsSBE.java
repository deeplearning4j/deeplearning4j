package org.deeplearning4j.optimizer.listener.stats;

import org.apache.commons.math3.exception.NumberIsTooLargeException;
import org.apache.commons.math3.exception.OutOfRangeException;
import org.deeplearning4j.optimize.listeners.stats.api.Histogram;
import org.deeplearning4j.optimize.listeners.stats.api.StatsType;
import org.deeplearning4j.optimize.listeners.stats.impl.SbeStatsInitializationReport;
import org.deeplearning4j.optimize.listeners.stats.impl.SbeStatsReport;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.factory.Nd4j;

import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * Created by Alex on 01/10/2016.
 */
public class TestStatsSBE {

    @Test
    public void testSbeStatsInitializationReport() {

        boolean[] tf = new boolean[]{true, false};

        //Hardware info
        int jvmAvailableProcessors = 1;
        int numDevices = 2;
        long jvmMaxMemory = 3;
        long offHeapMaxMemory = 4;
        long[] deviceTotalMemory = new long[]{5, 6};
        String[] deviceDescription = new String[]{"7", "8"};

        //Software info
        String arch = "9";
        String osName = "10";
        String jvmName = "11";
        String jvmVersion = "12";
        String jvmSpecVersion = "13";
        String nd4jBackendClass = "14";
        String nd4jDataTypeName = "15";

        //Model info
        String modelClassName = "16";
        String modelConfigJson = "17";
        String[] modelparamNames = new String[]{"18", "19", "20", "21"};
        int numLayers = 22;
        long numParams = 23;


        for (boolean hasHardwareInfo : tf) {
            for (boolean hasSoftwareInfo : tf) {
                for (boolean hasModelInfo : tf) {

                    SbeStatsInitializationReport report = new SbeStatsInitializationReport();
                    if (hasHardwareInfo) {
                        report.reportHardwareInfo(jvmAvailableProcessors, numDevices, jvmMaxMemory, offHeapMaxMemory, deviceTotalMemory, deviceDescription);
                    }

                    if (hasSoftwareInfo) {
                        report.reportSoftwareInfo(arch, osName, jvmName, jvmVersion, jvmSpecVersion, nd4jBackendClass, nd4jDataTypeName);
                    }

                    if (hasModelInfo) {
                        report.reportModelInfo(modelClassName, modelConfigJson, modelparamNames, numLayers, numParams);
                    }

                    byte[] asBytes = report.toByteArray();

                    SbeStatsInitializationReport report2 = new SbeStatsInitializationReport();
                    report2.fromByteArray(asBytes);

                    assertEquals(report, report2);

                    if (hasHardwareInfo) {
                        assertEquals(jvmAvailableProcessors, report2.getHwJvmAvailableProcessors());
                        assertEquals(numDevices, report2.getHwNumDevices());
                        assertEquals(jvmMaxMemory, report2.getHwJvmMaxMemory());
                        assertEquals(offHeapMaxMemory, report2.getHwOffHeapMaxMemory());
                        assertArrayEquals(deviceTotalMemory, report2.getHwDeviceTotalMemory());
                        assertArrayEquals(deviceDescription, report2.getHwDeviceDescription());
                    }
                    if (hasSoftwareInfo) {
                        assertEquals(arch, report2.getSwArch());
                        assertEquals(osName, report2.getSwOsName());
                        assertEquals(jvmName, report2.getSwJvmName());
                        assertEquals(jvmVersion, report2.getSwJvmVersion());
                        assertEquals(jvmSpecVersion, report2.getSwJvmSpecVersion());
                        assertEquals(nd4jBackendClass, report2.getSwNd4jBackendClass());
                        assertEquals(nd4jDataTypeName, report2.getSwNd4jDataTypeName());
                    }
                    if (hasModelInfo) {
                        assertEquals(modelClassName, report2.getModelClassName());
                        assertEquals(modelConfigJson, report2.getModelConfigJson());
                        assertArrayEquals(modelparamNames, report2.getModelParamNames());
                        assertEquals(numLayers, report2.getModelNumLayers());
                        assertEquals(numParams, report2.getModelNumParams());
                    }
                }
            }
        }
    }


    @Test
    public void testSbeStatsInitializationReportNullValues() {
        //Sanity check: shouldn't have any issues with encoding/decoding null values...
        boolean[] tf = new boolean[]{true, false};

        //Hardware info
        int jvmAvailableProcessors = 1;
        int numDevices = 2;
        long jvmMaxMemory = 3;
        long offHeapMaxMemory = 4;
        long[] deviceTotalMemory = null;
        String[] deviceDescription = null;

        //Software info
        String arch = null;
        String osName = null;
        String jvmName = null;
        String jvmVersion = null;
        String jvmSpecVersion = null;
        String nd4jBackendClass = null;
        String nd4jDataTypeName = null;

        //Model info
        String modelClassName = null;
        String modelConfigJson = null;
        String[] modelparamNames = null;
        int numLayers = 22;
        long numParams = 23;


        for (boolean hasHardwareInfo : tf) {
            for (boolean hasSoftwareInfo : tf) {
                for (boolean hasModelInfo : tf) {

                    SbeStatsInitializationReport report = new SbeStatsInitializationReport();
                    if (hasHardwareInfo) {
                        report.reportHardwareInfo(jvmAvailableProcessors, numDevices, jvmMaxMemory, offHeapMaxMemory, deviceTotalMemory, deviceDescription);
                    }

                    if (hasSoftwareInfo) {
                        report.reportSoftwareInfo(arch, osName, jvmName, jvmVersion, jvmSpecVersion, nd4jBackendClass, nd4jDataTypeName);
                    }

                    if (hasModelInfo) {
                        report.reportModelInfo(modelClassName, modelConfigJson, modelparamNames, numLayers, numParams);
                    }

                    byte[] asBytes = report.toByteArray();

                    SbeStatsInitializationReport report2 = new SbeStatsInitializationReport();
                    report2.fromByteArray(asBytes);

                    assertEquals(report, report2);

                    if (hasHardwareInfo) {
                        assertEquals(jvmAvailableProcessors, report2.getHwJvmAvailableProcessors());
                        assertEquals(numDevices, report2.getHwNumDevices());
                        assertEquals(jvmMaxMemory, report2.getHwJvmMaxMemory());
                        assertEquals(offHeapMaxMemory, report2.getHwOffHeapMaxMemory());
                        assertArrayEquals(deviceTotalMemory, report2.getHwDeviceTotalMemory());
                        assertArrayEquals(deviceDescription, report2.getHwDeviceDescription());
                    }
                    if (hasSoftwareInfo) {
                        assertEquals(arch, report2.getSwArch());
                        assertEquals(osName, report2.getSwOsName());
                        assertEquals(jvmName, report2.getSwJvmName());
                        assertEquals(jvmVersion, report2.getSwJvmVersion());
                        assertEquals(jvmSpecVersion, report2.getSwJvmSpecVersion());
                        assertEquals(nd4jBackendClass, report2.getSwNd4jBackendClass());
                        assertEquals(nd4jDataTypeName, report2.getSwNd4jDataTypeName());
                    }
                    if (hasModelInfo) {
                        assertEquals(modelClassName, report2.getModelClassName());
                        assertEquals(modelConfigJson, report2.getModelConfigJson());
                        assertArrayEquals(modelparamNames, report2.getModelParamNames());
                        assertEquals(numLayers, report2.getModelNumLayers());
                        assertEquals(numParams, report2.getModelNumParams());
                    }
                }
            }
        }
    }

    @Test
    public void testSbeStatsUpdate() {

        String[] paramNames = new String[]{"param0", "param1"};

        long time = System.currentTimeMillis();
        long duration = 123456;

        long perfRuntime = 1;
        long perfTotalEx = 2;
        long perfTotalMB = 3;
        double perfEPS = 4.0;
        double perfMBPS = 5.0;

        long memJC = 6;
        long memJM = 7;
        long memOC = 8;
        long memOM = 9;
        long[] memDC = new long[]{10,11};
        long[] memDM = new long[]{12,13};

        String gc1Name = "14";
        int gcd1 = 15;
        int gcdc1 = 16;
        int gcdt1 = 17;
        String gc2Name = "18";
        int gcd2 = 19;
        int gcdc2 = 20;
        int gcdt2 = 21;

        double score = 22.0;

        Map<String,Histogram> pHist = new HashMap<>();
        pHist.put(paramNames[0], new Histogram(23, 24, 2, new int[]{25,26}));
        pHist.put(paramNames[1], new Histogram(27, 28, 3, new int[]{29,30,31}));
        Map<String,Histogram> uHist = new HashMap<>();
        uHist.put(paramNames[0], new Histogram(32, 33, 2, new int[]{34, 35}));
        uHist.put(paramNames[1], new Histogram(36, 37, 3, new int[]{38, 39, 40}));
        Map<String,Histogram> aHist = new HashMap<>();
        aHist.put(paramNames[0], new Histogram(41, 42, 2, new int[]{43, 44}));
        aHist.put(paramNames[1], new Histogram(45, 46, 3, new int[]{47, 48, 47}));

        Map<String,Double> pMean = new HashMap<>();
        pMean.put(paramNames[0], 49.0);
        pMean.put(paramNames[1], 50.0);
        Map<String,Double> uMean = new HashMap<>();
        uMean.put(paramNames[0], 51.0);
        uMean.put(paramNames[1], 52.0);
        Map<String,Double> aMean = new HashMap<>();
        aMean.put(paramNames[0], 53.0);
        aMean.put(paramNames[1], 54.0);

        Map<String,Double> pStd = new HashMap<>();
        pStd.put(paramNames[0], 55.0);
        pStd.put(paramNames[1], 56.0);
        Map<String,Double> uStd = new HashMap<>();
        uStd.put(paramNames[0], 57.0);
        uStd.put(paramNames[1], 58.0);
        Map<String,Double> aStd = new HashMap<>();
        aStd.put(paramNames[0], 59.0);
        aStd.put(paramNames[1], 60.0);

        Map<String,Double> pMM = new HashMap<>();
        pMM.put(paramNames[0], 61.0);
        pMM.put(paramNames[1], 62.0);
        Map<String,Double> uMM = new HashMap<>();
        uMM.put(paramNames[0], 63.0);
        uMM.put(paramNames[1], 64.0);
        Map<String,Double> aMM = new HashMap<>();
        aMM.put(paramNames[0], 65.0);
        aMM.put(paramNames[1], 66.0);



        boolean[] tf = new boolean[]{true, false};
        boolean[][] tf3 = new boolean[][]{
                {false, false, false},
                {true, false, false},
                {false, true, false},
                {true, false, true},
                {true, true, true}};

        //Total tests: 2x2x2x2x2x2x
        for (boolean collectPerformanceStats : tf) {
            for (boolean collectMemoryStats : tf) {
                for (boolean collectGCStats : tf) {
                    for (boolean collectDataSetMetaData : tf) {
                        for (boolean collectScore : tf) {
                            for (boolean collectLearningRates : tf) {
                                for (boolean[] collectHistograms : tf3) {
                                    for (boolean[] collectMeanStdev : tf3) {
                                        for (boolean[] collectMM : tf3) {

                                            SbeStatsReport report = new SbeStatsReport();
                                            report.reportTime(time);
                                            report.reportStatsCollectionDurationMS(duration);
                                            if(collectPerformanceStats){
                                                report.reportPerformance(perfRuntime, perfTotalEx, perfTotalMB, perfEPS, perfMBPS);
                                            }

                                            if(collectMemoryStats){
                                                report.reportMemoryUse(memJC, memJM, memOC, memOM, memDC, memDM);
                                            }

                                            if(collectGCStats){
                                                report.reportGarbageCollection(gc1Name, gcd1, gcdc1, gcdt1);
                                                report.reportGarbageCollection(gc2Name, gcd2, gcdc2, gcdt2);
                                            }

                                            if(collectDataSetMetaData){
                                                //TODO
                                            }

                                            if(collectScore){
                                                report.reportScore(score);
                                            }

                                            if(collectLearningRates){
                                                //TODO
                                            }

                                            if(collectHistograms[0]){   //Param hist
                                                report.reportHistograms(StatsType.Parameters, pHist);
                                            }
                                            if(collectHistograms[1]){   //Update hist
                                                report.reportHistograms(StatsType.Updates, uHist);
                                            }
                                            if(collectHistograms[2]){   //Act hist
                                                report.reportHistograms(StatsType.Activations, aHist);
                                            }

                                            if(collectMeanStdev[0]){    //Param mean/stdev
                                                report.reportMean(StatsType.Parameters, pMean);
                                                report.reportStdev(StatsType.Parameters, pStd);
                                            }
                                            if(collectMeanStdev[1]){    //Update mean/stdev
                                                report.reportMean(StatsType.Updates, uMean);
                                                report.reportStdev(StatsType.Parameters, uStd);
                                            }
                                            if(collectMeanStdev[2]){    //Act mean/stdev
                                                report.reportMean(StatsType.Activations, aMean);
                                                report.reportStdev(StatsType.Parameters, aStd);
                                            }

                                            if(collectMM[0]){   //Param mean mag
                                                report.reportMeanMagnitudes(StatsType.Parameters, pMM);
                                            }
                                            if(collectMM[1]){   //Update mm
                                                report.reportMeanMagnitudes(StatsType.Updates, uMM);
                                            }
                                            if(collectMM[2]){   //Act mm
                                                report.reportMeanMagnitudes(StatsType.Activations, aMM);
                                            }


                                            byte[] bytes = report.toByteArray();



                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }


    }

}
