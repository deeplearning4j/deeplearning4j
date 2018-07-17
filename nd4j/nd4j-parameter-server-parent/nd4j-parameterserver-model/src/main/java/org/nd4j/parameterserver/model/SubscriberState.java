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

package org.nd4j.parameterserver.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.io.Serializable;
import java.util.Collections;
import java.util.Map;

/**
 * Reflects the state of
 * a parameter server
 * @author Adam Gibson
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class SubscriberState implements Serializable, Comparable<SubscriberState> {
    private boolean isMaster;
    private String serverState;
    private int totalUpdates;
    private int streamId;
    private String connectionInfo;
    private Map<String, Number> parameterUpdaterStatus;
    private boolean isAsync;
    private boolean isReady;



    /**
     * Returns an empty subscriber state
     * with -1 as total updates, master as false
     * and server state as empty
     * @return an empty subscriber state
     */
    public static SubscriberState empty() {
        return SubscriberState.builder().serverState("empty").streamId(-1)
                        .parameterUpdaterStatus(Collections.emptyMap()).totalUpdates(-1).isMaster(false).build();
    }



    /**
     * Write the subscriber state to the given {@link DataInput}
     * in the order of:
     * isMaster
     * serverState
     * totalUpdates
     * streamId
     * @param dataOutput the data output to write to
     * @throws IOException
     */
    public void write(DataOutput dataOutput) throws IOException {
        dataOutput.writeBoolean(isMaster);
        dataOutput.writeUTF(serverState);
        dataOutput.writeInt(totalUpdates);
        dataOutput.writeInt(streamId);

    }

    /**
     * Read the subscriber state to the given {@link DataInput}
     * in the order of:
     * isMaster
     * serverState
     * totalUpdates
     * streamId
     * @param dataInput the data output to write to
     * @throws IOException
     */
    public static SubscriberState read(DataInput dataInput) throws IOException {
        return SubscriberState.builder().isMaster(dataInput.readBoolean()).serverState(dataInput.readUTF())
                        .totalUpdates(dataInput.readInt()).streamId(dataInput.readInt()).build();
    }


    /**
     * Return the server opType (master or slave)
     * @return the server opType
     */
    public String serverType() {
        return isMaster ? "master" : "slave";
    }


    /**
     * Compares this object with the specified object for order.  Returns a
     * negative integer, zero, or a positive integer as this object is less
     * than, equal to, or greater than the specified object.
     * <p>
     * <p>The implementor must ensure <tt>sgn(x.compareTo(y)) ==
     * -sgn(y.compareTo(x))</tt> for all <tt>x</tt> and <tt>y</tt>.  (This
     * implies that <tt>x.compareTo(y)</tt> must throw an exception iff
     * <tt>y.compareTo(x)</tt> throws an exception.)
     * <p>
     * <p>The implementor must also ensure that the relation is transitive:
     * <tt>(x.compareTo(y)&gt;0 &amp;&amp; y.compareTo(z)&gt;0)</tt> implies
     * <tt>x.compareTo(z)&gt;0</tt>.
     * <p>
     * <p>Finally, the implementor must ensure that <tt>x.compareTo(y)==0</tt>
     * implies that <tt>sgn(x.compareTo(z)) == sgn(y.compareTo(z))</tt>, for
     * all <tt>z</tt>.
     * <p>
     * <p>It is strongly recommended, but <i>not</i> strictly required that
     * <tt>(x.compareTo(y)==0) == (x.equals(y))</tt>.  Generally speaking, any
     * class that implements the <tt>Comparable</tt> interface and violates
     * this condition should clearly indicate this fact.  The recommended
     * language is "Note: this class has a natural ordering that is
     * inconsistent with equals."
     * <p>
     * <p>In the foregoing description, the notation
     * <tt>sgn(</tt><i>expression</i><tt>)</tt> designates the mathematical
     * <i>signum</i> function, which is defined to return one of <tt>-1</tt>,
     * <tt>0</tt>, or <tt>1</tt> according to whether the value of
     * <i>expression</i> is negative, zero or positive.
     *
     * @param o the object to be compared.
     * @return a negative integer, zero, or a positive integer as this object
     * is less than, equal to, or greater than the specified object.
     * @throws NullPointerException if the specified object is null
     * @throws ClassCastException   if the specified object's opType prevents it
     *                              from being compared to this object.
     */
    @Override
    public int compareTo(SubscriberState o) {
        return Integer.compare(streamId, o.streamId);
    }
}
