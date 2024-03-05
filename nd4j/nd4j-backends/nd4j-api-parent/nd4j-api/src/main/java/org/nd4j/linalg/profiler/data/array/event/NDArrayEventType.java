/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.nd4j.linalg.profiler.data.array.event;

import org.nd4j.linalg.factory.Environment;

/**
 * The type Nd array event type.
 * This is used for differentiating between
 * different types of events that can occur
 * on an {@link org.nd4j.linalg.api.ndarray.INDArray}.
 * This is used in combination with
 * {@link Environment#isLogNDArrayEvents()}
 * and can be used to track changes to an {@link org.nd4j.linalg.api.ndarray.INDArray}
 * and track down difficult to understand problems
 * over the lifecycle of an {@link org.nd4j.linalg.api.ndarray.INDArray}.
 *
 * @author Adam Gibson
 */
public enum NDArrayEventType {

    //execution of a put
    PUT,
    //execution of an op where the op is an output
    OP_OUTPUT,
    //creation of a view of a given array
    VIEW_CREATION,
    //before execution of a put method on the array
    BEFORE_PUT,
    //before op execution where the op is an output
    BEFORE_OP_OUTPUT,
    //before creation of the ndarray s a view
    BEFORE_VIEW_CREATION,
    //after op execution where the op is an input
    OP_INPUT,
    //before op execution where the op is an input
    BEFORE_OP_INPUT,
    //close ndarray event
    CLOSE
    ,ARRAY_WORKSPACE_LEVERAGE,
    ARRAY_WORKSPACE_DETACH,
    ARRAY_CREATION;


    /**
     * Returns true if the given event type
     * has an after event
     * The following event types will have
     * after types:
     * {@link NDArrayEventType#BEFORE_OP_INPUT}
     * {@link NDArrayEventType#BEFORE_OP_OUTPUT}
     * {@link NDArrayEventType#BEFORE_PUT}
     * {@link NDArrayEventType#BEFORE_VIEW_CREATION}
     * @param eventType the event type to check
     * @return
     */
    public static boolean hasAfter(NDArrayEventType eventType) {
        switch (eventType) {
            case BEFORE_OP_INPUT:
            case BEFORE_OP_OUTPUT:
            case BEFORE_PUT:
            case BEFORE_VIEW_CREATION:
                return true;
            default:
                return false;
        }
    }

    /**
     * Returns the after type as denoted by
     * {@link #hasAfter(NDArrayEventType)}
     * This denotes the closing of an execution scope
     * reflecting the before and after state of an array
     * as well as the events in between.
     * @param eventType the event type to get the after type for
     * @return
     */
    public static NDArrayEventType afterFor(NDArrayEventType eventType) {
        switch (eventType) {
            case BEFORE_OP_INPUT:
                return OP_INPUT;
            case BEFORE_OP_OUTPUT:
                return OP_OUTPUT;
            case BEFORE_PUT:
                return PUT;
            case BEFORE_VIEW_CREATION:
                return VIEW_CREATION;
            default:
                throw new IllegalArgumentException("Illegal event type " + eventType);
        }
    }

}
