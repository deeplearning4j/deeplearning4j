package org.deeplearning4j.ui.i18n;

import org.deeplearning4j.ui.api.I18N;

/**
 * Returns the currently used I18N (Internationalization) class
 *
 * @author Alex Black
 */
public class I18NProvider {

    private static I18N i18n = DefaultI18N.getInstance();

    /**
     * Get the current/global I18N instance
     */
    public static I18N getInstance() {
        return i18n;
    }

}
