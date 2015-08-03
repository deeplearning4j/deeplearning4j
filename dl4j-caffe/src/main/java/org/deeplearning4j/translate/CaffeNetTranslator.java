package org.deeplearning4j.translate;

import org.deeplearning4j.caffe.Caffe.NetParameter;

import java.util.HashMap;
import java.util.Map;

/**
 * @author jeffreytang
 */
public class CaffeNetTranslator {

    public Map<String, String> netMappings;

    private void populateMappings() {
        netMappings = new HashMap<String, String>() {{
           put("", "");
        }};
    }

    public CaffeNetTranslator() { populateMappings(); }

    public NNCofigBuilderContainer translate(NetParameter net, NNCofigBuilderContainer builderContainer) {

        return builderContainer; // dummy
    }
}
