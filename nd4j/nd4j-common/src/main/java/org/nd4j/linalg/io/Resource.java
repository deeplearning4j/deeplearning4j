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

package org.nd4j.linalg.io;


import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.net.URL;

/**
 * Resource
 */
public interface Resource extends InputStreamSource {
    /**
     * Whether the resource exists on the classpath
     * @return
     */
    boolean exists();

    /**
     *
     * @return
     */
    boolean isReadable();

    /**
     *
     * @return
     */
    boolean isOpen();

    /**
     *
     * @return
     * @throws IOException
     */
    URL getURL() throws IOException;

    /**
     *
     * @return
     * @throws IOException
     */
    URI getURI() throws IOException;

    /**
     *
     * @return
     * @throws IOException
     */
    File getFile() throws IOException;

    /**
     *
     * @return
     * @throws IOException
     */
    long contentLength() throws IOException;

    /**
     *
     * @return
     * @throws IOException
     */
    long lastModified() throws IOException;

    /**
     *
     * @param var1
     * @return
     * @throws IOException
     */
    Resource createRelative(String var1) throws IOException;

    /**
     *
     * @return
     */
    String getFilename();

    /**
     *
     * @return
     */
    String getDescription();
}
