/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package com.atilika.kuromoji.util;

import java.io.IOException;
import java.io.InputStream;

/**
 * An adapter to resolve the required resources into data streams.
 */
public interface ResourceResolver {
    /**
     * Resolve the resource name and return an open input stream to it.
     *
     * @param resourceName resource to resolve
     * @return resolved resource stream
     * @throws IOException if an I/O error occured resolving the resource
     */
    InputStream resolve(String resourceName) throws IOException;
}
