/*-
 *
 *  * Copyright 2016 Skymind,Inc.
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
 */
package org.deeplearning4j.arbiter.util;

import org.slf4j.Logger;

import java.awt.*;
import java.net.URI;

/**
 * Various utilities for webpages and dealing with browsers
 */
public class WebUtils {

    public static void tryOpenBrowser(String path, Logger log) {
        try {
            WebUtils.openBrowser(new URI(path));
        } catch (Exception e) {
            log.error("Could not open browser", e);
            System.out.println("Browser could not be launched automatically.\nUI path: " + path);
        }
    }

    public static void openBrowser(URI uri) throws Exception {
        if (Desktop.isDesktopSupported()) {
            Desktop.getDesktop().browse(uri);
        } else {
            throw new UnsupportedOperationException(
                            "Cannot open browser on this platform: Desktop.isDesktopSupported() == false");
        }
    }

}
