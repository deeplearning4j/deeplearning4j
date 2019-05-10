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

package org.datavec.api.formats.input;


import org.datavec.api.conf.Configuration;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.InputSplit;
import org.datavec.api.writable.Writable;

import java.io.IOException;

/**
 * Create an input format
 *
 * @author Adam Gibson
 */
public interface InputFormat extends Writable {

    /**
     * Creates a reader from an input split
     * @param split the split to read
     * @return the reader from the given input split
     */
    RecordReader createReader(InputSplit split, Configuration conf) throws IOException, InterruptedException;

    /**
     * Creates a reader from an input split
     * @param split the split to read
     * @return the reader from the given input split
     */
    RecordReader createReader(InputSplit split) throws IOException, InterruptedException;

}
