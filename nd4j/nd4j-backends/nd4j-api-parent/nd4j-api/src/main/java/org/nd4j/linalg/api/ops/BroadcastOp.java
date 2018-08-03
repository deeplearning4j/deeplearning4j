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

package org.nd4j.linalg.api.ops;

/**
 * A broad cast op is one where a scalar
 * or less rank array
 * is broadcast to fill
 * a bigg er array.
 *
 * A typical broad cast operation would be adding a row to
 * each row in a matrix.
 *
 * @author Adam Gibson
 */
public interface BroadcastOp extends Op {

    /** Dimension to do the vector op along. Along dimension 1 for row vector ops,  along 0 for column vector ops */
    int[] getDimension();

    /** Set the dimension for the vector op. */
    void setDimension(int... dimension);


}
