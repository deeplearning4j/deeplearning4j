package org.nd4j.linalg.util;

import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;

/**
 *
 * Input stream utils
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
        return countLines(new BufferedInputStream(new FileInputStream(filename)));


    }
}