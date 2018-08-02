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
 * Simple pair implementation
 *
 * @author raver119@gmail.com
 */
@AllArgsConstructor
@Data
@Builder
public class ImmutablePair<K, V> implements Serializable {
    private static final long serialVersionUID = 119L;

    protected ImmutablePair() {
        //
    }

    @Setter(AccessLevel.NONE) protected K key;
    @Setter(AccessLevel.NONE) protected V value;

    public K getLeft() {
        return key;
    }

    public V getRight() {
        return value;
    }

    public K getFirst() {
        return key;
    }

    public V getSecond() {
        return value;
    }


    public static <T, E> ImmutablePair<T,E> of(T key, E value) {
        return new ImmutablePair<T, E>(key, value);
    }

    public static <T, E> ImmutablePair<T,E> makePair(T key, E value) {
        return new ImmutablePair<T, E>(key, value);
    }

    public static <T, E> ImmutablePair<T,E> create(T key, E value) {
        return new ImmutablePair<T, E>(key, value);
    }

    public static <T, E> ImmutablePair<T,E> pairOf(T key, E value) {
        return new ImmutablePair<T, E>(key, value);
    }
}
