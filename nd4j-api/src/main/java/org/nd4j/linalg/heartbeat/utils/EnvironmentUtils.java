package org.nd4j.linalg.heartbeat.utils;

import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.heartbeat.reports.Environment;

import java.net.InetAddress;
import java.net.NetworkInterface;
import java.security.SecureRandom;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * @author raver119@gmail.com
 */
public class EnvironmentUtils {

    /**
     * This method build
     * @return
     */
    public static Environment buildEnvironment() {
        Environment environment = new Environment();

        environment.setJavaVersion(System.getProperty("java.specification.version"));
        environment.setNumCores(Runtime.getRuntime().availableProcessors());
        environment.setAvailableMemory(Runtime.getRuntime().maxMemory());
        environment.setOsArch(System.getProperty("os.arch"));
        environment.setOsName(System.getProperty("os.name"));
        environment.setBackendUsed(Nd4j.getExecutioner().getClass().getSimpleName());

        return environment;
    }

    public static long buildCId() {
        /*
            builds repeatable anonymous value
        */
        long ret = 0;

        try {
            List<NetworkInterface> interfaces = Collections.list(NetworkInterface.getNetworkInterfaces());

            for (NetworkInterface networkInterface: interfaces) {
                try {
                    byte[] arr = networkInterface.getHardwareAddress();
                    long seed = 0;
                    for (int i = 0; i < arr.length; i++) {
                        seed += ((long) arr[i] & 0xffL) << (8 * i);
                    }
                    Random random = new Random(seed);

                    return random.nextLong();
                } catch (Exception e) {
                    ; // do nothing, just skip to next interface
                }
            }

        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        return ret ;
    }
}
