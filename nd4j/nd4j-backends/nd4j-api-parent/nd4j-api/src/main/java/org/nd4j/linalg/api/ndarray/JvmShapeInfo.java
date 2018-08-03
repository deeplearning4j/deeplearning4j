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

package org.nd4j.linalg.api.ndarray;


import lombok.Getter;
import lombok.NonNull;
import org.nd4j.linalg.api.shape.Shape;

public class JvmShapeInfo {
    @Getter protected long[] javaShapeInformation;
    @Getter protected long[] shape;
    @Getter protected long[] stride;
    @Getter protected long length;
    @Getter protected long ews;
    @Getter protected long extras;
    @Getter protected char order;
    @Getter protected int rank;

    public JvmShapeInfo(@NonNull long[] javaShapeInformation) {
        this.javaShapeInformation = javaShapeInformation;
        this.shape = Shape.shape(javaShapeInformation);
        this.stride = Shape.stride(javaShapeInformation);
        this.length = Shape.length(javaShapeInformation);
        this.ews = Shape.elementWiseStride(javaShapeInformation);
        this.extras = Shape.extras(javaShapeInformation);
        this.order = Shape.order(javaShapeInformation);
        this.rank = Shape.rank(javaShapeInformation);
    }
}
