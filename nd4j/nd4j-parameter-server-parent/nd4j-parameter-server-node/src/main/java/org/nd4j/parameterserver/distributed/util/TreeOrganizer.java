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

package org.nd4j.parameterserver.distributed.util;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;

import java.io.Serializable;
import java.util.Collection;

/**
 * This class provides methods for ephemeral tree management
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class TreeOrganizer implements Serializable {
    private static final long serialVersionUID = 1L;

    /**
     * This methods adds new node to the tree
     */
    public void addNode() {
        //
    }

    /**
     * This method removes  node from tree
     */
    public void removeNode() {
        //
    }


    /**
     * This method returns true, if node is known
     * @return
     */
    public boolean isKnownNode() {
        return false;
    }


    /**
     * This method reconnects given node to another node
     */
    public void remapNode() {
        //
    }

    /**
     * This method returns upstream connection for a given node
     */
    public void getUpstreamForNode() {
        //
    }

    /**
     * This method returns downstream connections for a given node
     */
    public void getDownstreamForNode() {
        //
    }

    /**
     * This method returns total number of nodes below given one
     * @return
     */
    public int numberOfDescendantsOfNode() {
        return 0;
    }

    /**
     * This method returns Node representing given Id
     * @return
     */
    protected Node getNodeById() {
        return null;
    }

    /**
     * This method returns Node representing given IP
     * @return
     */
    protected Node getNodeByIp() {
        return null;
    }

    /**
     * This class represents basic tree node
     */
    @NoArgsConstructor
    @AllArgsConstructor
    public static class Node implements Serializable {
        private static final long serialVersionUID = 1L;

        @NonNull private String id;
        @NonNull private String ip;
        @NonNull private int port;

        private Node upstream;

        private Collection<Node> downstream;
    }
}
