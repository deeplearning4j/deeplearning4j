package org.deeplearning4j.ui.i18n;

import org.deeplearning4j.ui.api.I18N;

/**
 * Created by Alex on 14/10/2016.
 */
public class I18NProvider {

    private static I18N i18n = DefaultI18N.getInstance();

    public static I18N getInstance(){
        return i18n;
    }

}
