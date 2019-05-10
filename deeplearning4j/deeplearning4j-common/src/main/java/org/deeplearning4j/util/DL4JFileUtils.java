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

package org.deeplearning4j.util;

import org.deeplearning4j.config.DL4JSystemProperties;

import java.io.File;
import java.io.IOException;

/**
 * Utilities for working with temporary files
 *
 * @author Alex Black
 */
public class DL4JFileUtils {

    private DL4JFileUtils(){ }

    /**
     * Create a temporary file in the location specified by {@link DL4JSystemProperties#DL4J_TEMP_DIR_PROPERTY} if set,
     * or the default temporary directory (usually specified by java.io.tmpdir system property)
     * @param prefix Prefix for generating file's name; must be at least 3 characeters
     * @param suffix Suffix for generating file's name; may be null (".tmp" will be used if null)
     * @return A temporary file
     */
    public static File createTempFile(String prefix, String suffix) {
        String p = System.getProperty(DL4JSystemProperties.DL4J_TEMP_DIR_PROPERTY);
        try {
            if (p == null || p.isEmpty()) {
                return File.createTempFile(prefix, suffix);
            } else {
                return File.createTempFile(prefix, suffix, new File(p));
            }
        } catch (IOException e){
            throw new RuntimeException("Error creating temporary file", e);
        }
    }

}
