package org.deeplearning4j.translate;

import java.util.Map;

/**
 * @author jeffreytang
 */
public interface CaffeTranslatorInterface {

    void specialTranslation(String caffeField,
                            Object caffeVal,
                            String builderField,
                            Map<String, Object> builderParamMap);

}
