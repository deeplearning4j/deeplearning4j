package org.deeplearning4j.caffe.translate;

import java.util.Map;

/**
 * @author jeffreytang
 */
public interface CaffeSpecialTranslator {

    void specialTranslation(String caffeField,
                            Object caffeVal,
                            String builderField,
                            Map<String, Object> builderParamMap);
}
