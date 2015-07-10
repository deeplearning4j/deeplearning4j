package translate;

import com.google.gson.JsonObject;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by jeffreytang on 7/6/15.
 */
public class ParseCaffeLogic {

    public static Map parseSolver(Map caffeSovlerMap) {
        // The JsonObject containing all  the useful fields in Solver
        HashMap<String, Object> translatedSolverMap = new HashMap<String, Object>();

        // HashMap for solver parameter mapping from Caffe to DL4J
        HashMap<String, String> caffeToDL4JSolver = new HashMap<String, String>();
        caffeToDL4JSolver.put("lr", "baseLr");
        caffeToDL4JSolver.put("averageLoss", "averageLoss_");
        caffeToDL4JSolver.put("maxIter", "maxIter_");
        caffeToDL4JSolver.put("iterSize", "iterSize_");
        caffeToDL4JSolver.put("lrPolicy", "lrPolicy_");
        caffeToDL4JSolver.put("gamma", "gamma_");
        caffeToDL4JSolver.put("power", "power_");
        caffeToDL4JSolver.put("momentum", "momentum_");
        caffeToDL4JSolver.put("weightDecay", "weightDecay_");
        caffeToDL4JSolver.put("regularization", "regularizationType_");
        caffeToDL4JSolver.put("stepSize", "stepsize_");
        caffeToDL4JSolver.put("backEnd", "solverMode_");
        caffeToDL4JSolver.put("deviceId", "deviceId_");
        caffeToDL4JSolver.put("randomSeed", "randomSeed_");
        caffeToDL4JSolver.put("solverType", "solverType_");
        caffeToDL4JSolver.put("delta", "delta_");

        for (Map.Entry<String, String> entry : caffeToDL4JSolver.entrySet()) {
            String dL4JKey = entry.getKey();
            String caffeKey = entry.getValue();

            translatedSolverMap.put(dL4JKey, caffeSovlerMap.get(caffeKey));
        }
        return translatedSolverMap;
    }

    public static Object caffeToDL4JlayerConversion(String caffeLayerType) {
        // Define the layer mapping between Caffe and DL4J
        Map<String, Object> caffeToDL4JLayerMapping = new HashMap<String, Object>();
        caffeToDL4JLayerMapping.put("Data", null);
        caffeToDL4JLayerMapping.put("Data", null);
        caffeToDL4JLayerMapping.put("Data", null);
        caffeToDL4JLayerMapping.put("Data", null);
        caffeToDL4JLayerMapping.put("Data", null);

        return caffeToDL4JLayerMapping.get(caffeLayerType);
    }

//    public static Map parseNet(Map caffeNetMap) {
//        // The JsonObject containing all  the useful fields in Solver
//        HashMap<String, Object> translatedNetMap = new HashMap<String, Object>();
//
//        // Loop through all the layers and extract the relevant parameters
//        ArrayList layerArrayList = (ArrayList) caffeNetMap.get("layer_");
//        for (Object layerObject : layerArrayList) {
//            Map layerMap = (Map) layerObject;
//            String layerType = (String) layerMap.get("type_");
//            String layerName = (String) layerMap.get("name_");
//            ArrayList topLayers = (ArrayList) layerMap.get("top_");
//            ArrayList bottomLayers = (ArrayList) layerMap.get("bottom_");
//
//
//        }
//
//
//    }
}
