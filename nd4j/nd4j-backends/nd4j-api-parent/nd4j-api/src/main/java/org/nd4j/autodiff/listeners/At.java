/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.autodiff.listeners;

import lombok.*;
import org.nd4j.autodiff.samediff.internal.FrameIter;

/**
 *
 * Used in SameDiff {@link Listener} instances.
 * Contains information such as the current epoch, iteration and thread
 *
 * @author Alex Black
 */
@AllArgsConstructor
@EqualsAndHashCode
@ToString
@Builder
@Setter
public class At {

    private int epoch;
    private int iteration;
    private int trainingThreadNum;
    private long javaThreadNum;
    private FrameIter frameIter;
    private Operation operation;

    /**
     * @return A new instance with everything set to 0, and operation set to INFERENCE
     */
    public static At defaultAt(){
        return new At(0, 0, 0, 0, null, Operation.INFERENCE);
    }

    /**
     * @param op Operation
     * @return A new instance with everything set to 0, except for the specified operation
     */
    public static At defaultAt(@NonNull Operation op){
        return new At(0, 0, 0, 0, null, op);
    }

    /**
     * @return The current training epoch
     */
    public int epoch(){
        return epoch;
    }

    /**
     * @return The current training iteration
     */
    public int iteration(){
        return iteration;
    }

    /**
     * @return The number of the SameDiff thread
     */
    public int trainingThreadNum(){
        return trainingThreadNum;
    }

    /**
     * @return The Java/JVM thread number for training
     */
    public long javaThreadNum(){
        return javaThreadNum;
    }

    /**
     * @return The current operation
     */
    public Operation operation(){
        return operation;
    }

    /**
     * @return A copy of the current At instance
     */
    public At copy(){
        return new At(epoch, iteration, trainingThreadNum, javaThreadNum, frameIter, operation);
    }

    /**
     * @param operation Operation to set in the new instance
     * @return A copy of the current instance, but with the specified operation
     */
    public At copy(Operation operation){
        return new At(epoch, iteration, trainingThreadNum, javaThreadNum, frameIter, operation);
    }
}
