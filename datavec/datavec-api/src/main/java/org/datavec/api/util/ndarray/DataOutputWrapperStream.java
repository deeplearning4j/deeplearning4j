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

package org.datavec.api.util.ndarray;

import lombok.AllArgsConstructor;

import java.io.DataOutput;
import java.io.IOException;
import java.io.OutputStream;

/**
 * A simple class to wrap a {@link DataOutput} instance in an {@link OutputStream}
 *
 * @author Alex Black
 */
@AllArgsConstructor
public class DataOutputWrapperStream extends OutputStream {

    private DataOutput underlying;

    @Override
    public void write(int b) throws IOException {
        //write(int) method: "Writes to the output stream the eight low-order bits of the argument b. The 24 high-order
        // bits of b are ignored."
        underlying.write(b);
    }
}
