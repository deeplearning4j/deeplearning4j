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

package org.nd4j.autodiff.samediff.internal.memory;

import lombok.NonNull;
import org.nd4j.autodiff.samediff.internal.SessionMemMgr;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Abstract memory manager, that implements ulike and dup methods using the underlying allocate methods
 *
 * @author Alex Black
 */
public abstract class AbstractMemoryMgr implements SessionMemMgr {

    @Override
    public INDArray ulike(@NonNull INDArray arr) {
        return allocate(false, arr.dataType(), arr.shape());
    }

    @Override
    public INDArray dup(@NonNull INDArray arr) {
        INDArray out = ulike(arr);
        out.assign(arr);
        return out;
    }
}
