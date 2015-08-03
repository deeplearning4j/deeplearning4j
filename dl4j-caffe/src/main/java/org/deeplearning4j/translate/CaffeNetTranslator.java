package org.deeplearning4j.translate;

import org.deeplearning4j.caffe.Caffe.NetParameter;

import java.util.HashMap;
import java.util.Map;

/**
 * @author jeffreytang
 */
public class CaffeNetTranslator {

    public Map<String, String> relevantLayerMappings;
    public Map<String, String> netParamMappings;

    private void fillRelevantLayerMappings() {
        relevantLayerMappings = new HashMap<String, String>() {{
            put("", "");
        }};
    }

    private void fillNetParamMappings() {
        netParamMappings = new HashMap<String, String>() {{
           put("", "");
        }};
    }


    public CaffeNetTranslator() {
        fillNetParamMappings();
        fillRelevantLayerMappings();
    }

    public NNCofigBuilderContainer translate(NetParameter net, NNCofigBuilderContainer builderContainer) {

        return builderContainer; // dummy
    }
}
