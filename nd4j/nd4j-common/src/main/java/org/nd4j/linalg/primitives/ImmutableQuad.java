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

package org.nd4j.linalg.primitives;

import lombok.*;

import java.io.Serializable;

/**
 * Simple quad elements holder implementation
 * @author raver119@gmail.com
 */
@Data
@AllArgsConstructor
@Builder
public class ImmutableQuad<F, S, T, O> implements Serializable {
    private static final long serialVersionUID = 119L;

    @Setter(AccessLevel.NONE) protected F first;
    @Setter(AccessLevel.NONE) protected S second;
    @Setter(AccessLevel.NONE) protected T third;
    @Setter(AccessLevel.NONE) protected O fourth;

    public static <F, S, T, O> ImmutableQuad<F, S, T, O> quadOf(F first, S second, T third, O fourth) {
        return new ImmutableQuad(first, second, third, fourth);
    }

    public static <F, S, T, O> ImmutableQuad<F, S,T, O> of(F first, S second, T third, O fourth) {
        return new ImmutableQuad(first, second, third, fourth);
    }
}
