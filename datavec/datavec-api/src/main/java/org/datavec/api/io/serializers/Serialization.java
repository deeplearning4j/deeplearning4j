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

package org.datavec.api.io.serializers;

/**
 * <p>
 * Encapsulates a {@link Serializer}/{@link Deserializer} pair.
 * </p>
 * @param <T>
 */
public interface Serialization<T> {

    /**
     * Allows clients to test whether this {@link Serialization}
     * supports the given class.
     */
    boolean accept(Class<?> c);

    /**
     * @return a {@link Serializer} for the given class.
     */
    Serializer<T> getSerializer(Class<T> c);

    /**
     * @return a {@link Deserializer} for the given class.
     */
    Deserializer<T> getDeserializer(Class<T> c);


}
