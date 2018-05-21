package org.deeplearning4j.ui.play.staticroutes;

import org.deeplearning4j.ui.i18n.I18NProvider;
import play.mvc.Result;

import java.util.function.Function;

import static play.mvc.Results.ok;

/**
 * Route for global internationalization setting
 *
 * @author Alex Black
 */
public class I18NRoute implements Function<String, Result> {
    @Override
    public Result apply(String s) {
        I18NProvider.getInstance().setDefaultLanguage(s);
        return ok();
    }
}
