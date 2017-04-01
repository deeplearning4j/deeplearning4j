/*-
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 *
 */

package org.nd4j.linalg.util;

import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;

/**
 * Input stream jcuda.utils
 *
 * @author Adam Gibson
 */
public class InputStreamUtil {
    /**
     * Count number of lines in a file
     *
     * @param is
     * @return
     * @throws IOException
     */
    public static int countLines(InputStream is) throws IOException {
        try {
            byte[] c = new byte[1024];
            int count = 0;
            int readChars = 0;
            boolean empty = true;
            while ((readChars = is.read(c)) != -1) {
                empty = false;
                for (int i = 0; i < readChars; ++i) {
                    if (c[i] == '\n') {
                        ++count;
                    }
                }
            }
            return (count == 0 && !empty) ? 1 : count;
        } finally {
            is.close();
        }


    }

    /**
     * Count number of lines in a file
     *
     * @param filename
     * @return
     * @throws IOException
     */
    public static int countLines(String filename) throws IOException {
        FileInputStream fis = new FileInputStream(filename);
        try {
            return countLines(new BufferedInputStream(fis));
        } finally {
            fis.close();
        }
    }
}
