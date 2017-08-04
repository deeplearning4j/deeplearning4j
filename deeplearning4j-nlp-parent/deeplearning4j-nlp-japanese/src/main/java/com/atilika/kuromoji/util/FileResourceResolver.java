/*-*
 * Copyright © 2010-2015 Atilika Inc. and contributors (see CONTRIBUTORS.md)
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may
 * not use this file except in compliance with the License.  A copy of the
 * License is distributed with this work in the LICENSE.md file.  You may
 * also obtain a copy of the License from
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
                e.printStackTrace();
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
