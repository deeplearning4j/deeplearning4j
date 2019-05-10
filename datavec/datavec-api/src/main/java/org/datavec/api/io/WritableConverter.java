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

package org.datavec.api.io;

import org.datavec.api.io.converters.WritableConverterException;
import org.datavec.api.writable.Writable;

/**
 * Convert a writable to another writable (used for say: transitioning dates or categorical to numbers)
 *
 * @author Adam Gibson
 */
public interface WritableConverter {


    /**
     * Convert a writable to another kind of writable
     * @param writable the writable to convert
     * @return the converted writable
     */
    Writable convert(Writable writable) throws WritableConverterException;

}
