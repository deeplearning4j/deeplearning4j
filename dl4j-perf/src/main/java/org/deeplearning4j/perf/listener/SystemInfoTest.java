package org.deeplearning4j.perf.listener;

/**
 * Oshi (https://github.com/oshi/oshi)
 *
 * Copyright (c) 2010 - 2018 The Oshi Project Team
 *
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 *
 * Maintainers:
 * dblock[at]dblock[dot]org
 * widdis[at]gmail[dot]com
 * enrico.bianchi[at]gmail[dot]com
 *
 * Contributors:
 * https://github.com/oshi/oshi/graphs/contributors
 */

import java.util.Arrays;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import oshi.SystemInfo;
import oshi.hardware.Baseboard;
import oshi.hardware.CentralProcessor;
import oshi.hardware.CentralProcessor.TickType;
import oshi.hardware.ComputerSystem;
import oshi.hardware.Display;
import oshi.hardware.Firmware;
import oshi.hardware.GlobalMemory;
import oshi.hardware.HWDiskStore;
import oshi.hardware.HWPartition;
import oshi.hardware.HardwareAbstractionLayer;
import oshi.hardware.NetworkIF;
import oshi.hardware.PowerSource;
import oshi.hardware.Sensors;
import oshi.hardware.UsbDevice;
import oshi.software.os.FileSystem;
import oshi.software.os.NetworkParams;
import oshi.software.os.OSFileStore;
import oshi.software.os.OSProcess;
import oshi.software.os.OperatingSystem;
import oshi.software.os.OperatingSystem.ProcessSort;
import oshi.util.FormatUtil;
import oshi.util.Util;

/**
 * The Class SystemInfoTest.
 *
 * @author dblock[at]dblock[dot]org
 */
public class SystemInfoTest {

    /**
     * The main method.
     *
     * @param args
     *            the arguments
     */
    public static void main(String[] args) {
        // Options: ERROR > WARN > INFO > DEBUG > TRACE
        Logger LOG = LoggerFactory.getLogger(SystemInfoTest.class);

        LOG.info("Initializing System...");
        SystemInfo si = new SystemInfo();

        HardwareAbstractionLayer hal = si.getHardware();
        OperatingSystem os = si.getOperatingSystem();

        System.out.println(os);

        LOG.info("Checking computer system...");
        printComputerSystem(hal.getComputerSystem());

        LOG.info("Checking Processor...");
        printProcessor(hal.getProcessor());

        LOG.info("Checking Memory...");
        printMemory(hal.getMemory());

        LOG.info("Checking CPU...");
        printCpu(hal.getProcessor());

        LOG.info("Checking Processes...");
        printProcesses(os, hal.getMemory());

        LOG.info("Checking Sensors...");
        printSensors(hal.getSensors());

        LOG.info("Checking Power sources...");
        printPowerSources(hal.getPowerSources());

        LOG.info("Checking Disks...");
        printDisks(hal.getDiskStores());

        LOG.info("Checking File System...");
        printFileSystem(os.getFileSystem());

        LOG.info("Checking Network interfaces...");
        printNetworkInterfaces(hal.getNetworkIFs());

        LOG.info("Checking Network parameterss...");
        printNetworkParameters(os.getNetworkParams());

        // hardware: displays
        LOG.info("Checking Displays...");
        printDisplays(hal.getDisplays());

        // hardware: USB devices
        LOG.info("Checking USB Devices...");
        printUsbDevices(hal.getUsbDevices(true));
    }

    private static void printComputerSystem(final ComputerSystem computerSystem) {

        System.out.println("manufacturer: " + computerSystem.getManufacturer());
        System.out.println("model: " + computerSystem.getModel());
        System.out.println("serialnumber: " + computerSystem.getSerialNumber());
        final Firmware firmware = computerSystem.getFirmware();
        System.out.println("firmware:");
        System.out.println("  manufacturer: " + firmware.getManufacturer());
        System.out.println("  name: " + firmware.getName());
        System.out.println("  description: " + firmware.getDescription());
        System.out.println("  version: " + firmware.getVersion());
        System.out.println("  release date: " + (firmware.getReleaseDate() == null ? "unknown"
                : firmware.getReleaseDate() == null ? "unknown" : FormatUtil.formatDate(firmware.getReleaseDate())));
        final Baseboard baseboard = computerSystem.getBaseboard();
        System.out.println("baseboard:");
        System.out.println("  manufacturer: " + baseboard.getManufacturer());
        System.out.println("  model: " + baseboard.getModel());
        System.out.println("  version: " + baseboard.getVersion());
        System.out.println("  serialnumber: " + baseboard.getSerialNumber());
    }

    private static void printProcessor(CentralProcessor processor) {
        System.out.println(processor);
        System.out.println(" " + processor.getPhysicalProcessorCount() + " physical CPU core(s)");
        System.out.println(" " + processor.getLogicalProcessorCount() + " logical CPU(s)");

        System.out.println("Identifier: " + processor.getIdentifier());
        System.out.println("ProcessorID: " + processor.getProcessorID());
    }

    private static void printMemory(GlobalMemory memory) {
        System.out.println("Memory: " + FormatUtil.formatBytes(memory.getAvailable()) + "/"
                + FormatUtil.formatBytes(memory.getTotal()));
        System.out.println("Swap used: " + FormatUtil.formatBytes(memory.getSwapUsed()) + "/"
                + FormatUtil.formatBytes(memory.getSwapTotal()));
    }

    private static void printCpu(CentralProcessor processor) {
        System.out.println("Uptime: " + FormatUtil.formatElapsedSecs(processor.getSystemUptime()));

        long[] prevTicks = processor.getSystemCpuLoadTicks();
        System.out.println("CPU, IOWait, and IRQ ticks @ 0 sec:" + Arrays.toString(prevTicks));
        // Wait a second...
        Util.sleep(1000);
        long[] ticks = processor.getSystemCpuLoadTicks();
        System.out.println("CPU, IOWait, and IRQ ticks @ 1 sec:" + Arrays.toString(ticks));
        long user = ticks[TickType.USER.getIndex()] - prevTicks[TickType.USER.getIndex()];
        long nice = ticks[TickType.NICE.getIndex()] - prevTicks[TickType.NICE.getIndex()];
        long sys = ticks[TickType.SYSTEM.getIndex()] - prevTicks[TickType.SYSTEM.getIndex()];
        long idle = ticks[TickType.IDLE.getIndex()] - prevTicks[TickType.IDLE.getIndex()];
        long iowait = ticks[TickType.IOWAIT.getIndex()] - prevTicks[TickType.IOWAIT.getIndex()];
        long irq = ticks[TickType.IRQ.getIndex()] - prevTicks[TickType.IRQ.getIndex()];
        long softirq = ticks[TickType.SOFTIRQ.getIndex()] - prevTicks[TickType.SOFTIRQ.getIndex()];
        long totalCpu = user + nice + sys + idle + iowait + irq + softirq;

        System.out.format(
                "User: %.1f%% Nice: %.1f%% System: %.1f%% Idle: %.1f%% IOwait: %.1f%% IRQ: %.1f%% SoftIRQ: %.1f%%",
                100d * user / totalCpu, 100d * nice / totalCpu, 100d * sys / totalCpu, 100d * idle / totalCpu,
                100d * iowait / totalCpu, 100d * irq / totalCpu, 100d * softirq / totalCpu);
        System.out.format("CPU load: %.1f%% (counting ticks)%n", processor.getSystemCpuLoadBetweenTicks() * 100);
        System.out.format("CPU load: %.1f%% (OS MXBean)%n", processor.getSystemCpuLoad() * 100);
        double[] loadAverage = processor.getSystemLoadAverage(3);
        System.out.println("CPU load averages:" + (loadAverage[0] < 0 ? " N/A" : String.format(" %.2f", loadAverage[0]))
                + (loadAverage[1] < 0 ? " N/A" : String.format(" %.2f", loadAverage[1]))
                + (loadAverage[2] < 0 ? " N/A" : String.format(" %.2f", loadAverage[2])));
        // per core CPU
        StringBuilder procCpu = new StringBuilder("CPU load per processor:");
        double[] load = processor.getProcessorCpuLoadBetweenTicks();
        for (double avg : load) {
            procCpu.append(String.format(" %.1f%%", avg * 100));
        }
        System.out.println(procCpu.toString());
    }

    private static void printProcesses(OperatingSystem os, GlobalMemory memory) {
        System.out.println("Processes: " + os.getProcessCount() + ", Threads: " + os.getThreadCount());
        // Sort by highest CPU
        List<OSProcess> procs = Arrays.asList(os.getProcesses(5, ProcessSort.CPU));

        System.out.println("   PID  %CPU %MEM       VSZ       RSS Name");
        for (int i = 0; i < procs.size() && i < 5; i++) {
            OSProcess p = procs.get(i);
            System.out.format(" %5d %5.1f %4.1f %9s %9s %s%n", p.getProcessID(),
                    100d * (p.getKernelTime() + p.getUserTime()) / p.getUpTime(),
                    100d * p.getResidentSetSize() / memory.getTotal(), FormatUtil.formatBytes(p.getVirtualSize()),
                    FormatUtil.formatBytes(p.getResidentSetSize()), p.getName());
        }
    }

    private static void printSensors(Sensors sensors) {
        System.out.println("Sensors:");
        System.out.format(" CPU Temperature: %.1f°C%n", sensors.getCpuTemperature());
        System.out.println(" Fan Speeds: " + Arrays.toString(sensors.getFanSpeeds()));
        System.out.format(" CPU Voltage: %.1fV%n", sensors.getCpuVoltage());
    }

    private static void printPowerSources(PowerSource[] powerSources) {
        StringBuilder sb = new StringBuilder("Power: ");
        if (powerSources.length == 0) {
            sb.append("Unknown");
        } else {
            double timeRemaining = powerSources[0].getTimeRemaining();
            if (timeRemaining < -1d) {
                sb.append("Charging");
            } else if (timeRemaining < 0d) {
                sb.append("Calculating time remaining");
            } else {
                sb.append(String.format("%d:%02d remaining", (int) (timeRemaining / 3600),
                        (int) (timeRemaining / 60) % 60));
            }
        }
        for (PowerSource pSource : powerSources) {
            sb.append(String.format("%n %s @ %.1f%%", pSource.getName(), pSource.getRemainingCapacity() * 100d));
        }
        System.out.println(sb.toString());
    }

    private static void printDisks(HWDiskStore[] diskStores) {
        System.out.println("Disks:");
        for (HWDiskStore disk : diskStores) {
            boolean readwrite = disk.getReads() > 0 || disk.getWrites() > 0;
            System.out.format(" %s: (model: %s - S/N: %s) size: %s, reads: %s (%s), writes: %s (%s), xfer: %s ms%n",
                    disk.getName(), disk.getModel(), disk.getSerial(),
                    disk.getSize() > 0 ? FormatUtil.formatBytesDecimal(disk.getSize()) : "?",
                    readwrite ? disk.getReads() : "?", readwrite ? FormatUtil.formatBytes(disk.getReadBytes()) : "?",
                    readwrite ? disk.getWrites() : "?", readwrite ? FormatUtil.formatBytes(disk.getWriteBytes()) : "?",
                    readwrite ? disk.getTransferTime() : "?");
            HWPartition[] partitions = disk.getPartitions();
            if (partitions == null) {
                // TODO Remove when all OS's implemented
                continue;
            }
            for (HWPartition part : partitions) {
                System.out.format(" |-- %s: %s (%s) Maj:Min=%d:%d, size: %s%s%n", part.getIdentification(),
                        part.getName(), part.getType(), part.getMajor(), part.getMinor(),
                        FormatUtil.formatBytesDecimal(part.getSize()),
                        part.getMountPoint().isEmpty() ? "" : " @ " + part.getMountPoint());
            }
        }
    }

    private static void printFileSystem(FileSystem fileSystem) {
        System.out.println("File System:");

        System.out.format(" File Descriptors: %d/%d%n", fileSystem.getOpenFileDescriptors(),
                fileSystem.getMaxFileDescriptors());

        OSFileStore[] fsArray = fileSystem.getFileStores();
        for (OSFileStore fs : fsArray) {
            long usable = fs.getUsableSpace();
            long total = fs.getTotalSpace();
            System.out.format(
                    " %s (%s) [%s] %s of %s free (%.1f%%) is %s "
                            + (fs.getVolume() != null && fs.getVolume().length() > 0 ? "[%s]" : "%s")
                            + " and is mounted at %s%n",
                    fs.getName(), fs.getDescription().isEmpty() ? "file system" : fs.getDescription(), fs.getType(),
                    FormatUtil.formatBytes(usable), FormatUtil.formatBytes(fs.getTotalSpace()), 100d * usable / total,
                    fs.getVolume(), fs.getVolume(), fs.getMount());
        }
    }

    private static void printNetworkInterfaces(NetworkIF[] networkIFs) {
        System.out.println("Network interfaces:");
        for (NetworkIF net : networkIFs) {
            System.out.format(" Name: %s (%s)%n", net.getName(), net.getDisplayName());
            System.out.format("   MAC Address: %s %n", net.getMacaddr());
            System.out.format("   MTU: %s, Speed: %s %n", net.getMTU(), FormatUtil.formatValue(net.getSpeed(), "bps"));
            System.out.format("   IPv4: %s %n", Arrays.toString(net.getIPv4addr()));
            System.out.format("   IPv6: %s %n", Arrays.toString(net.getIPv6addr()));
            boolean hasData = net.getBytesRecv() > 0 || net.getBytesSent() > 0 || net.getPacketsRecv() > 0
                    || net.getPacketsSent() > 0;
            System.out.format("   Traffic: received %s/%s%s; transmitted %s/%s%s %n",
                    hasData ? net.getPacketsRecv() + " packets" : "?",
                    hasData ? FormatUtil.formatBytes(net.getBytesRecv()) : "?",
                    hasData ? " (" + net.getInErrors() + " err)" : "",
                    hasData ? net.getPacketsSent() + " packets" : "?",
                    hasData ? FormatUtil.formatBytes(net.getBytesSent()) : "?",
                    hasData ? " (" + net.getOutErrors() + " err)" : "");
        }
    }

    private static void printNetworkParameters(NetworkParams networkParams) {
        System.out.println("Network parameters:");
        System.out.format(" Host name: %s%n", networkParams.getHostName());
        System.out.format(" Domain name: %s%n", networkParams.getDomainName());
        System.out.format(" DNS servers: %s%n", Arrays.toString(networkParams.getDnsServers()));
        System.out.format(" IPv4 Gateway: %s%n", networkParams.getIpv4DefaultGateway());
        System.out.format(" IPv6 Gateway: %s%n", networkParams.getIpv6DefaultGateway());
    }

    private static void printDisplays(Display[] displays) {
        System.out.println("Displays:");
        int i = 0;
        for (Display display : displays) {
            System.out.println(" Display " + i + ":");
            System.out.println(display.toString());
            i++;
        }
    }

    private static void printUsbDevices(UsbDevice[] usbDevices) {
        System.out.println("USB Devices:");
        for (UsbDevice usbDevice : usbDevices) {
            System.out.println(usbDevice.toString());
        }
    }
}