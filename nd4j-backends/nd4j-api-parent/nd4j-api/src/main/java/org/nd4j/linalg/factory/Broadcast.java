/*-
 *
 *  * Copyright 2017 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 *
 */

package org.nd4j.linalg.factory;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.broadcast.*;

/**
 * Convenience methods for broadcasts
 *
 * @author Alex Black
 */
public class Broadcast {

    private Broadcast(){ }

    /**
     * Broadcast add op. See: {@link BroadcastAddOp}
     */
    public static INDArray add(INDArray x, INDArray y, INDArray z, int... dimensions ){
        return Nd4j.getExecutioner().execAndReturn(new BroadcastAddOp(x,y,z,dimensions));
    }

    /**
     * Broadcast copy op. See: {@link BroadcastCopyOp}
     */
    public static INDArray copy(INDArray x, INDArray y, INDArray z, int... dimensions ){
        return Nd4j.getExecutioner().execAndReturn(new BroadcastCopyOp(x,y,z,dimensions));
    }

    /**
     * Broadcast divide op. See: {@link BroadcastDivOp}
     */
    public static INDArray div(INDArray x, INDArray y, INDArray z, int... dimensions ){
        return Nd4j.getExecutioner().execAndReturn(new BroadcastDivOp(x,y,z,dimensions));
    }

    /**
     * Broadcast equal to op. See: {@link BroadcastEqualTo}
     */
    public static INDArray eq(INDArray x, INDArray y, INDArray z, int... dimensions ){
        return Nd4j.getExecutioner().execAndReturn(new BroadcastEqualTo(x,y,z,dimensions));
    }

    /**
     * Broadcast greater than op. See: {@link BroadcastGreaterThan}
     */
    public static INDArray gt(INDArray x, INDArray y, INDArray z, int... dimensions ){
        return Nd4j.getExecutioner().execAndReturn(new BroadcastGreaterThan(x,y,z,dimensions));
    }

    /**
     * Broadcast greater than or equal to op. See: {@link BroadcastGreaterThanOrEqual}
     */
    public static INDArray gte(INDArray x, INDArray y, INDArray z, int... dimensions ){
        return Nd4j.getExecutioner().execAndReturn(new BroadcastGreaterThanOrEqual(x,y,z,dimensions));
    }

    /**
     * Broadcast less than op. See: {@link BroadcastLessThan}
     */
    public static INDArray lt(INDArray x, INDArray y, INDArray z, int... dimensions ){
        return Nd4j.getExecutioner().execAndReturn(new BroadcastLessThan(x,y,z,dimensions));
    }

    /**
     * Broadcast less than or equal to op. See: {@link BroadcastLessThanOrEqual}
     */
    public static INDArray lte(INDArray x, INDArray y, INDArray z, int... dimensions ){
        return Nd4j.getExecutioner().execAndReturn(new BroadcastLessThanOrEqual(x,y,z,dimensions));
    }

    /**
     * Broadcast element-wise multiply op. See: {@link BroadcastMulOp}
     */
    public static INDArray mul(INDArray x, INDArray y, INDArray z, int... dimensions ){
        return Nd4j.getExecutioner().execAndReturn(new BroadcastMulOp(x,y,z,dimensions));
    }

    /**
     * Broadcast not equal to op. See: {@link BroadcastNotEqual}
     */
    public static INDArray neq(INDArray x, INDArray y, INDArray z, int... dimensions ){
        return Nd4j.getExecutioner().execAndReturn(new BroadcastNotEqual(x,y,z,dimensions));
    }

    /**
     * Broadcast reverse division op. See: {@link BroadcastRDivOp}
     */
    public static INDArray rdiv(INDArray x, INDArray y, INDArray z, int... dimensions ){
        return Nd4j.getExecutioner().execAndReturn(new BroadcastRDivOp(x,y,z,dimensions));
    }

    /**
     * Broadcast reverse subtraction op. See: {@link BroadcastRSubOp}
     */
    public static INDArray rsub(INDArray x, INDArray y, INDArray z, int... dimensions ){
        return Nd4j.getExecutioner().execAndReturn(new BroadcastRSubOp(x,y,z,dimensions));
    }

    /**
     * Broadcast subtraction op. See: {@link BroadcastSubOp}
     */
    public static INDArray sub(INDArray x, INDArray y, INDArray z, int... dimensions ){
        return Nd4j.getExecutioner().execAndReturn(new BroadcastSubOp(x,y,z,dimensions));
    }

    /**
     * Broadcast max op. See: {@link BroadcastMax}
     */
    public static INDArray max(INDArray x, INDArray y, INDArray z, int... dimensions ){
        return Nd4j.getExecutioner().execAndReturn(new BroadcastMax(x,y,z,dimensions));
    }

    /**
     * Broadcast min op. See: {@link BroadcastMin}
     */
    public static INDArray min(INDArray x, INDArray y, INDArray z, int... dimensions ){
        return Nd4j.getExecutioner().execAndReturn(new BroadcastMin(x,y,z,dimensions));
    }

    /**
     * Broadcast absolute max op. See: {@link BroadcastAMax}
     */
    public static INDArray amax(INDArray x, INDArray y, INDArray z, int... dimensions ){
        return Nd4j.getExecutioner().execAndReturn(new BroadcastAMax(x,y,z,dimensions));
    }

    /**
     * Broadcast absolute min op. See: {@link BroadcastAMax}
     */
    public static INDArray amin(INDArray x, INDArray y, INDArray z, int... dimensions ){
        return Nd4j.getExecutioner().execAndReturn(new BroadcastAMin(x,y,z,dimensions));
    }
}
