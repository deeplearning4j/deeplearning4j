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

package org.nd4j.primitives;

import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.Serializer;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;
import org.nd4j.linalg.primitives.AtomicDouble;

/**
 * Serializer for AtomicDouble (needs a serializer due to long field being transient...)
 */
public class AtomicDoubleSerializer extends Serializer<AtomicDouble> {
    @Override
    public void write(Kryo kryo, Output output, AtomicDouble a) {
        output.writeDouble(a.get());
    }

    @Override
    public AtomicDouble read(Kryo kryo, Input input, Class<AtomicDouble> a) {
        return new AtomicDouble(input.readDouble());
    }
}
