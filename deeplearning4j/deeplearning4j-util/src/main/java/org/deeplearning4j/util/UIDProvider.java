package org.deeplearning4j.util;

import lombok.extern.slf4j.Slf4j;

import java.net.NetworkInterface;
import java.rmi.server.UID;
import java.util.Enumeration;

/**
 * Static methods for obtaining unique identifiers for both the machine (hardware) and the JVM.
 *
 * Note: the unique hardware identifier does NOT provide any strong guarantees of uniqueness of the returned identifier
 * with respect to machine restarts and hardware changes, and should not be relied upon for anything where guarantees
 * are required.
 * Note also that as a fallback, if no hardware UID can be determined, the JVM UID will be returned as the hardware UID also.
 *
 * @author Alex Black
 */
@Slf4j
public class UIDProvider {

    private static final String JVM_UID;
    private static final String HARDWARE_UID;

    static {

        UID jvmUIDSource = new UID();
        String asString = jvmUIDSource.toString();
        //Format here: hexStringFromRandomNumber:hexStringFromSystemClock:hexStringOfUIDInstance
        //The first two components here will be identical for all UID instances in a JVM, where as the 'hexStringOfUIDInstance'
        // will vary (increment) between UID object instances. So we'll only be using the first two components here
        int lastIdx = asString.lastIndexOf(":");
        JVM_UID = asString.substring(0, lastIdx).replaceAll(":", "");


        //Assumptions here:
        //1. getNetworkInterfaces() returns at least one non-null element
        //   This is guaranteed by getNetworkInterfaces() Javadoc: "The {@code Enumeration} contains at least one element..."
        //2. That the iteration order for network interfaces is consistent between JVM instances on the same hardware
        //   This appears to hold, but no formal guarantees seem to be available here
        //3. That MAC addresses are 'unique enough' for our purposes
        byte[] address = null;
        boolean noInterfaces = false;
        Enumeration<NetworkInterface> niEnumeration = null;
        try {
            niEnumeration = NetworkInterface.getNetworkInterfaces();
        } catch (Exception e) {
            noInterfaces = true;
        }

        if (niEnumeration != null) {
            while (niEnumeration.hasMoreElements()) {
                NetworkInterface ni = niEnumeration.nextElement();
                byte[] addr;
                try {
                    addr = ni.getHardwareAddress();
                } catch (Exception e) {
                    continue;
                }
                if (addr == null || addr.length != 6)
                    continue; //May be null (if it can't be obtained) or not standard 6 byte MAC-48 representation

                address = addr;
                break;
            }
        }

        if (address == null) {
            log.warn("Could not generate hardware UID{}. Using fallback: JVM UID as hardware UID.",
                            (noInterfaces ? " (no interfaces)" : ""));
            HARDWARE_UID = JVM_UID;
        } else {
            StringBuilder sb = new StringBuilder();
            for (byte b : address) {
                sb.append(String.format("%02x", b));
            }
            HARDWARE_UID = sb.toString();
        }
    }

    private UIDProvider() {}


    public static String getJVMUID() {
        return JVM_UID;
    }

    public static String getHardwareUID() {
        return HARDWARE_UID;
    }



}
