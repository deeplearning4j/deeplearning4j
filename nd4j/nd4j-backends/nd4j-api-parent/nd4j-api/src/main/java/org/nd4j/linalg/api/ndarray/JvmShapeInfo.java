/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
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

package org.nd4j.linalg.api.ndarray;


import lombok.Getter;
import lombok.NonNull;
import org.nd4j.linalg.api.shape.Shape;

public class JvmShapeInfo {
    @Getter protected final long[] javaShapeInformation;
    @Getter protected final long[] shape;
    @Getter protected final long[] stride;
    @Getter protected final long length;
    @Getter protected final long ews;
    @Getter protected final long extras;
    @Getter protected final char order;
    @Getter protected final int rank;

    public JvmShapeInfo(@NonNull long[] javaShapeInformation) {
        this.javaShapeInformation = javaShapeInformation;
        this.shape = Shape.shape(javaShapeInformation);
        this.stride = Shape.stride(javaShapeInformation);
        this.length = Shape.isEmpty(javaShapeInformation) ? 0 : Shape.length(javaShapeInformation);
        this.ews = Shape.elementWiseStride(javaShapeInformation);
        this.extras = Shape.extras(javaShapeInformation);
        this.order = Shape.order(javaShapeInformation);
        this.rank = Shape.rank(javaShapeInformation);
    }
}
