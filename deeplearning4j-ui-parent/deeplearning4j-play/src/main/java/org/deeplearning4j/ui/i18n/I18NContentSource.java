package org.deeplearning4j.ui.i18n;

/**
 * Created by Alex on 13/10/2016.
 */
public interface I18NContentSource {

    boolean hasKey(String lang, String key);

    String getMessage(String lang, String key);

}
