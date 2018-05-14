package org.deeplearning4j.perf.listener;

import lombok.Builder;
import lombok.Data;
import org.nd4j.linalg.api.environment.Nd4jEnvironment;
import org.nd4j.linalg.api.ops.performance.PerformanceTracker;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.memory.MemcpyDirection;
import oshi.json.SystemInfo;
import oshi.json.hardware.CentralProcessor;
import oshi.json.hardware.GlobalMemory;
import oshi.json.hardware.HWDiskStore;
import oshi.json.software.os.NetworkParams;
import oshi.util.Util;

import java.io.Serializable;
import java.util.*;

@Builder
@Data
public class HardwareMetric implements Serializable {

    private Map<Integer,DeviceMetric> perCoreMetrics;
    private long physicalProcessorCount,logicalProcessorCount;
    private long currentMemoryUse;
    private Map<Integer,DeviceMetric> gpuMetrics;
    private String hostName;
    private long ioWaitTime;
    private long averagedCpuLoad;
    private Map<Integer,DiskInfo> diskInfo;
    private String name;


    /**
     * Runs {@link #fromSystem(SystemInfo)}
     * with a fresh {@link SystemInfo}
     * @return the hardware metric based on
     * the current snapshot of the system this
     * runs on
     */
    public static HardwareMetric fromSystem() {
        return fromSystem(new SystemInfo());
    }



    /**
     * Returns the relevant information
     * needed for system diagnostics
     * based on the {@link SystemInfo}
     * @param systemInfo the system info to use
     * @return the {@link HardwareMetric} for the
     * system this process runs on
     */
    public static HardwareMetric fromSystem(SystemInfo systemInfo) {
        return fromSystem(systemInfo,UUID.randomUUID().toString());
    }

    /**
     * Returns the relevant information
     * needed for system diagnostics
     * based on the {@link SystemInfo}
     * @param systemInfo the system info to use
     * @return the {@link HardwareMetric} for the
     * system this process runs on
     */
    public static HardwareMetric fromSystem(SystemInfo systemInfo,String name) {
        HardwareMetricBuilder builder = HardwareMetric.builder();
        CentralProcessor processor = systemInfo.getHardware().getProcessor();
        long[] prevTicks = processor.getSystemCpuLoadTicks();
        // Wait a second...
        Util.sleep(1000);
        long[] ticks = processor.getSystemCpuLoadTicks();
        long iowait = ticks[oshi.hardware.CentralProcessor.TickType.IOWAIT.getIndex()] - prevTicks[oshi.hardware.CentralProcessor.TickType.IOWAIT.getIndex()];

        GlobalMemory globalMemory = systemInfo.getHardware().getMemory();
        NetworkParams networkParams = systemInfo.getOperatingSystem().getNetworkParams();

        double[] processorCpuLoadBetweenTicks = processor.getProcessorCpuLoadBetweenTicks();
        Map<Integer,DeviceMetric> cpuMetrics = new LinkedHashMap<>();
        for(int i = 0; i < processorCpuLoadBetweenTicks.length; i++) {
            cpuMetrics.put(i, DeviceMetric.builder()
                    .load(processorCpuLoadBetweenTicks[i]).
                            build());
        }


        Map<Integer,DiskInfo> diskInfoMap = new LinkedHashMap<>();

        HWDiskStore[] diskStores = systemInfo.getHardware().getDiskStores();
        for(int i = 0; i < diskStores.length; i++) {
            HWDiskStore diskStore = diskStores[i];
            DiskInfo diskInfo = DiskInfo.builder()
                    .bytesRead(diskStore.getReadBytes())
                    .bytesWritten(diskStore.getWriteBytes())
                    .name(diskStore.getName())
                    .modelName(diskStore.getModel())
                    .transferTime(diskStore.getTransferTime())
                    .build();
            diskInfoMap.put(i,diskInfo);

        }

        Map<Integer,DeviceMetric> gpuMetric = new HashMap<>();
        if(Nd4j.getBackend().getClass().getName().toLowerCase().contains("cublas")) {
            Properties info = Nd4j.getExecutioner().getEnvironmentInformation();
            /**
             *
             */

            List<Map<String, Object>> devicesList = (List<Map<String, Object>>) info.get(Nd4jEnvironment.CUDA_DEVICE_INFORMATION_KEY);
            for(int i = 0; i < devicesList.size(); i++) {
                double available = Double.parseDouble(devicesList.get(i).get(Nd4jEnvironment.CUDA_FREE_MEMORY_KEY).toString());
                Map<MemcpyDirection, Long> memcpyDirectionLongMap = PerformanceTracker.getInstance().getCurrentBandwidth().get(i);
                DeviceMetric deviceMetric = DeviceMetric.builder()
                        .bandwidthHostToDevice(memcpyDirectionLongMap.get(MemcpyDirection.HOST_TO_DEVICE))
                        .bandwidthDeviceToHost(memcpyDirectionLongMap.get(MemcpyDirection.DEVICE_TO_HOST))
                        .bandwidthDeviceToDevice(memcpyDirectionLongMap.get(MemcpyDirection.DEVICE_TO_DEVICE))
                        .memAvailable(available).totalMemory(Double.parseDouble(devicesList.get(i).get(Nd4jEnvironment.CUDA_TOTAL_MEMORY_KEY).toString()))
                        .deviceName(devicesList.get(i).get(Nd4jEnvironment.CUDA_DEVICE_NAME_KEY).toString())
                        .build();
                gpuMetric.put(i,deviceMetric);

            }
        }

        return builder.logicalProcessorCount(processor.getLogicalProcessorCount())
                .physicalProcessorCount(processor.getPhysicalProcessorCount())
                .name(name)
                .averagedCpuLoad((long) processor.getSystemCpuLoad() * 100)
                .ioWaitTime(iowait).gpuMetrics(gpuMetric)
                .hostName(networkParams.getHostName()).diskInfo(diskInfoMap)
                .currentMemoryUse(globalMemory.getTotal() - globalMemory.getAvailable())
                .perCoreMetrics(cpuMetrics)
                .build();


    }

}
