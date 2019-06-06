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

package org.datavec.api.records.reader.factory;

import org.datavec.api.exceptions.UnknownFormatException;
import org.datavec.api.records.reader.RecordReader;

import java.net.URI;

/**
 * Factory for creating RecordReader instance
 *
 * @author sonali
 */
public interface RecordReaderFactory {
    /**
     * Creates instance of RecordReader
     *
     * @param uri
     * @return record reader instance
     * @throws UnknownFormatException
     */
    RecordReader create(URI uri) throws UnknownFormatException;
}
