package org.deeplearning4j.ui.api;

import java.awt.*;

/**
 * Created by Alex on 4/04/2016.
 */
public class Utils {

    public static String colorToHex(Color color){
        return String.format("#%02x%02x%02x", color.getRed(), color.getGreen(), color.getBlue());
    }

}
