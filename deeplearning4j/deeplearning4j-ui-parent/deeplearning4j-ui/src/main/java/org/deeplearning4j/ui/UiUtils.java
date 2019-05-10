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

package org.deeplearning4j.ui;

import org.slf4j.Logger;

import java.awt.*;
import java.net.URI;

public class UiUtils {

    private UiUtils() {}

    public static void tryOpenBrowser(String path, Logger log) {
        try {
            UiUtils.openBrowser(new URI(path));
        } catch (Exception e) {
            //   log.error("Could not open browser",e);
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
