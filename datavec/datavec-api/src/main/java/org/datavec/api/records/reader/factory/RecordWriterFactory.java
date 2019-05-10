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

import org.datavec.api.records.writer.RecordWriter;

import java.net.URI;

/**
 * Factory for creating RecordWriter instance
 *
 * @author sonali
 */
public interface RecordWriterFactory {

    /**
     *
     * @param uri destination for saving model
     * @return record writer instance
     * @throws Exception
     */

    RecordWriter create(URI uri) throws Exception;
}
