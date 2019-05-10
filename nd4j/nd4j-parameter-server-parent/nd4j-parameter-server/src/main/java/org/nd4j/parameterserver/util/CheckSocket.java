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

package org.nd4j.parameterserver.util;

import java.io.IOException;
import java.net.*;

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
    public static boolean remotePortTaken(String node, int port, int timeout) {
        Socket s = null;
        try {
            s = new Socket();
            s.setReuseAddress(true);
            SocketAddress sa = new InetSocketAddress(node, port);
            s.connect(sa, timeout * 1000);
        } catch (IOException e) {
            if (e.getMessage().equals("Connection refused")) {
                return false;
            }
            if (e instanceof SocketTimeoutException || e instanceof UnknownHostException) {
                throw e;
            }
        } finally {
            if (s != null) {
                if (s.isConnected()) {
                    return true;
                } else {
                }
                try {
                    s.close();
                } catch (IOException e) {
                }
            }

            return false;
        }

    }
}
