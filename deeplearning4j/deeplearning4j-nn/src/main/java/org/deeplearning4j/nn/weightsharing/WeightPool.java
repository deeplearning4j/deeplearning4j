package org.deeplearning4j.nn.weightsharing;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import org.nd4j.linalg.api.ndarray.INDArray;

public class WeightPool {
    public Map<String, INDArray> params;
    public INDArray paramsFlattened;

    public Map<String,INDArray> weightNoiseParams = new HashMap<>();

    private static Map<String, WeightPool> pools = new HashMap<>();
    private static Set<String> usedIds = new HashSet<>();

    public static String getNewId(){
        String id;

        int i = 20;
        int j = 0;

        do{
            String SALTCHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890";
            StringBuilder salt = new StringBuilder();
            Random rnd = new Random();
            while (salt.length() < i) { // length of the random string.
                int index = (int) (rnd.nextFloat() * SALTCHARS.length());
                salt.append(SALTCHARS.charAt(index));
            }
            id = salt.toString();

            j++;
            if(j == 100){
                i++;
                j = 0;
            }

        } while(usedIds.contains(id));

        return id;
    }

    public static WeightPool getOrCreatePool(String id){
        if(pools.containsKey(id)){
            return pools.get(id);
        } else {
            pools.put(id, new WeightPool());
            return pools.get(id);
        }
    }

    public static RecurrentWeightPool getOrCreateRecurrentPool(String id){
        if(pools.containsKey(id)){
            return (RecurrentWeightPool) pools.get(id);
        } else {
            pools.put(id, new RecurrentWeightPool());
            return (RecurrentWeightPool) pools.get(id);
        }
    }
}
