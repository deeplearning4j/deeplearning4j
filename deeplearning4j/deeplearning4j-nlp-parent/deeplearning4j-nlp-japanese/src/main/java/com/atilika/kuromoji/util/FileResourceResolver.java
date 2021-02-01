/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;

public class FileResourceResolver implements ResourceResolver {
    protected static final Logger log = LoggerFactory.getLogger(FileResourceResolver.class);

    static {
        if (KuromojiBinFilesFetcher.kuromojiExist() == false) {
            log.info("Kuromoji bin folder not exist ");
            try {
                KuromojiBinFilesFetcher.downloadAndUntar();
            } catch (IOException e) {
                log.error("IOException : ", e);
            }
        }
    }

    public FileResourceResolver() {}

    @Override
    public InputStream resolve(String fileName) throws IOException {
        InputStream input = new FileInputStream(new File(fileName));
        if (input == null) {
            throw new IOException("Classpath resource not found: " + fileName);
        }
        return input;
    }
}
