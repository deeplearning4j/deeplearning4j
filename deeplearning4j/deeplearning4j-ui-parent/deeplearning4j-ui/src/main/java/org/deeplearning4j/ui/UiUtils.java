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
