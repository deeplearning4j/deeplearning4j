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
package org.nd4j.linalg.profiler.data.array.registry;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.nd4j.common.primitives.AtomicBoolean;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class NDArrayWithContext implements Serializable {
    private StackTraceElement[] context;
    private String array;
    private long originalId;
    private static AtomicBoolean callingFromContext = new AtomicBoolean(false);

    public static NDArrayWithContext from(INDArray array) {
        if(callingFromContext.get())
            return null;
        callingFromContext.set(true);
        NDArrayWithContext ret =  builder()
                .array(array.toStringFull())
                .originalId(array.getId())
                .context(Thread.currentThread().getStackTrace())
                .build();
        callingFromContext.set(false);
        return ret;
    }
}
