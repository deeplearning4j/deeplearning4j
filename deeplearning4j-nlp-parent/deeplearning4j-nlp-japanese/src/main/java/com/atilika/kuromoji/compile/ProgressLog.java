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
package com.atilika.kuromoji.compile;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;

/**
 * Simple progress logger
 */
public class ProgressLog {
    private static int indent = 0;
    private static boolean atEOL = false;
    private static DateFormat dateFormat = new SimpleDateFormat("HH:mm:ss");
    private static Map<Integer, Long> startTimes = new HashMap<>();

    public static void begin(String message) {
        newLine();
        System.out.print(leader() + message + "... ");
        System.out.flush();
        atEOL = true;
        indent++;
        startTimes.put(indent, System.currentTimeMillis());
    }

    public static void end() {
        newLine();
        Long start = startTimes.get(indent);
        indent = Math.max(0, indent - 1);
        System.out.println(leader() + "done"
                        + (start != null ? " [" + ((System.currentTimeMillis() - start) / 1000) + "s]" : ""));
        System.out.flush();
    }

    public static void println(String message) {
        newLine();
        System.out.println(leader() + message);
        System.out.flush();
    }

    private static void newLine() {
        if (atEOL) {
            System.out.println();
        }
        atEOL = false;
    }

    private static String leader() {
        return "[KUROMOJI] " + dateFormat.format(new Date()) + ": "
                        + (new String(new char[indent * 4]).replace("\0", " "));
    }
}
