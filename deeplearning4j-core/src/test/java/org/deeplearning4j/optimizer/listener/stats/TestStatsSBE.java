package org.deeplearning4j.optimizer.listener.stats;

import org.apache.commons.math3.exception.NumberIsTooLargeException;
import org.apache.commons.math3.exception.OutOfRangeException;
import org.deeplearning4j.optimize.listeners.stats.impl.SbeStatsInitializationReport;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;

/**
 * Created by Alex on 01/10/2016.
 */
public class TestStatsSBE {

    @Test
    public void testSbeStatsInitializationReport(){

        boolean[] tf = new boolean[]{true,false};

//        //Hardware info
//        int jvmAvailableProcessors = 1;
//        int numDevices = 2;
//        long jvmMaxMemory = 3;
//        long offHeapMaxMemory = 4;
//        long[] deviceTotalMemory = new long[]{5,6};
//        String[] deviceDescription = new String[]{"7","8"};
//
//        //Software info
//        String arch = "9";
//        String osName = "10";
//        String jvmName = "11";
//        String jvmVersion = "12";
//        String jvmSpecVersion = "13";
//        String nd4jBackendClass = "14";
//        String nd4jDataTypeName = "15";
//
//        //Model info
//        String modelClassName = "16";
//        String modelConfigJson = "17";
//        String[] modelparamNames = new String[]{"18","19","20","21"};
//        int numLayers = 22;
//        long numParams = 23;



        //Hardware info
        int jvmAvailableProcessors = 1;
        int numDevices = 2;
        long jvmMaxMemory = 3;
        long offHeapMaxMemory = 4;
        long[] deviceTotalMemory = new long[]{5,6};
        String[] deviceDescription = new String[]{"",""};

        //Software info
        String arch = "";
        String osName = "";
        String jvmName = "";
        String jvmVersion = "";
        String jvmSpecVersion = "";
        String nd4jBackendClass = "";
        String nd4jDataTypeName = "";

        //Model info
        String modelClassName = "";
        String modelConfigJson = "";
        String[] modelparamNames = new String[]{"18","19","20","21"};
        int numLayers = 22;
        long numParams = 23;


        for(boolean hasHardwareInfo : tf){
            for(boolean hasSoftwareInfo : tf){
                for(boolean hasModelInfo : tf){

                    SbeStatsInitializationReport report = new SbeStatsInitializationReport();
                    if(hasHardwareInfo){
                        report.reportHardwareInfo(jvmAvailableProcessors, numDevices, jvmMaxMemory, offHeapMaxMemory, deviceTotalMemory, deviceDescription);
                    }

                    if(hasSoftwareInfo){
                        report.reportSoftwareInfo(arch, osName, jvmName, jvmVersion, jvmSpecVersion, nd4jBackendClass, nd4jDataTypeName);
                    }

                    if(hasModelInfo){
                        report.reportModelInfo(modelClassName, modelConfigJson, modelparamNames, numLayers, numParams);
                    }

                    byte[] asBytes = report.toByteArray();

                    SbeStatsInitializationReport report2 = new SbeStatsInitializationReport();
                    report2.fromByteArray(asBytes);

                    assertEquals(report, report2);


                }
            }
        }


    }

}
