package org.deeplearning4j.ui.api;

import java.awt.*;

public class Utils {

    private Utils() {}

    /** Convert an AWT color to a hex color string, such as #000000 */
    public static String colorToHex(Color color) {
        return String.format("#%02x%02x%02x", color.getRed(), color.getGreen(), color.getBlue());
    }

}
