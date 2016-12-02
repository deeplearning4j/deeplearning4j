package org.nd4j.parameterserver.util;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.net.SocketAddress;
import java.net.SocketTimeoutException;
import java.net.UnknownHostException;

/**
 * Credit: http://stackoverflow.com/questions/5226905/test-if-remote-port-is-in-use
 *
 *
 */
public class CheckSocket {

    /**
     * Check if a remote port is taken
     * @param node the host to check
     * @param port the port to check
     * @param timeout the timeout for the connection
     * @return true if the port is taken false otherwise
     */
    public static boolean remotePortTaken(String node,int port,int timeout) {
        Socket s = null;
        try {
            s = new Socket();
            s.setReuseAddress(true);
            SocketAddress sa = new InetSocketAddress(node, port);
            s.connect(sa, timeout * 1000);
        } catch (IOException e) {
            if ( e.getMessage().equals("Connection refused")) {
                return false;
            }
            if (e instanceof SocketTimeoutException ||  e instanceof UnknownHostException) {
                throw e;
            }
        } finally {
            if (s != null) {
                if ( s.isConnected()) {
                    return false;
                } else {
                }
                try {
                    s.close();
                } catch (IOException e) {
                }
            }

            return true;
        }

    }
}