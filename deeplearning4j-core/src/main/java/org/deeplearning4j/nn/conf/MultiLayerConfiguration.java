package org.deeplearning4j.nn.conf;

import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.module.SimpleModule;
import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.nn.conf.deserializers.ActivationFunctionDeSerializer;
import org.deeplearning4j.nn.conf.deserializers.DistributionDeSerializer;
import org.deeplearning4j.nn.conf.deserializers.RandomGeneratorDeSerializer;
import org.deeplearning4j.nn.conf.serializers.ActivationFunctionSerializer;
import org.deeplearning4j.nn.conf.serializers.DistributionSerializer;
import org.deeplearning4j.nn.conf.serializers.RandomGeneratorSerializer;
import org.nd4j.linalg.api.activation.ActivationFunction;

import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Configuration for a multi layer network
 *
 * @author Adam Gibson
 */
public class MultiLayerConfiguration implements Serializable {

    private int[] hiddenLayerSizes;
    private List<NeuralNetConfiguration> confs;

    private MultiLayerConfiguration() {}



    public NeuralNetConfiguration getConf(int i) {
        return confs.get(i);
    }

    public int[] getHiddenLayerSizes() {
        return hiddenLayerSizes;
    }

    public void setHiddenLayerSizes(int[] hiddenLayerSizes) {
        this.hiddenLayerSizes = hiddenLayerSizes;
    }

    public List<NeuralNetConfiguration> getConfs() {
        return confs;
    }

    public void setConfs(List<NeuralNetConfiguration> confs) {
        this.confs = confs;
    }

    /**
     * Object mapper for serialization of configurations
     * @return
     */
    public static ObjectMapper mapper() {
        ObjectMapper ret = new ObjectMapper();
        ret.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);

        SimpleModule module = new SimpleModule();
        module.addDeserializer(ActivationFunction.class,new ActivationFunctionDeSerializer());
        module.addSerializer(ActivationFunction.class, new ActivationFunctionSerializer());
        module.addDeserializer(RandomGenerator.class, new RandomGeneratorDeSerializer());
        module.addSerializer(RandomGenerator.class, new RandomGeneratorSerializer());
        module.addSerializer(RealDistribution.class, new DistributionSerializer());
        module.addDeserializer(RealDistribution.class, new DistributionDeSerializer());
        ret.registerModule(module);
        return ret;
    }

    /**
     *
     * @return
     */
    public String toJson() {
        ObjectMapper mapper = mapper();
        try {
            return mapper.writeValueAsString(this).replaceAll("\"activationFunction\",","").replaceAll("\"rng\",","").replaceAll("\"dist\",", "");
        } catch (com.fasterxml.jackson.core.JsonProcessingException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Create a neural net configuration from json
     * @param json the neural net configuration from json
     * @return
     */
    public static MultiLayerConfiguration fromJson(String json) {
        ObjectMapper mapper = mapper();
        try {
            return mapper.readValue(json, MultiLayerConfiguration.class);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof MultiLayerConfiguration)) return false;

        MultiLayerConfiguration that = (MultiLayerConfiguration) o;

        if (confs != null ? !confs.equals(that.confs) : that.confs != null) return false;
        if (!Arrays.equals(hiddenLayerSizes, that.hiddenLayerSizes)) return false;

        return true;
    }

    @Override
    public int hashCode() {
        int result = hiddenLayerSizes != null ? Arrays.hashCode(hiddenLayerSizes) : 0;
        result = 31 * result + (confs != null ? confs.hashCode() : 0);
        return result;
    }

    public static class Builder {

        private List<NeuralNetConfiguration> confs = new ArrayList<>();
        private int[] hiddenLayerSizes;

        public Builder confs(List<NeuralNetConfiguration> confs) {
            this.confs = confs;
            return this;

        }


        public Builder hiddenLayerSizes(int[] hiddenLayerSizes) {
            this.hiddenLayerSizes = hiddenLayerSizes;
            return this;
        }

        public MultiLayerConfiguration build() {
            MultiLayerConfiguration conf = new MultiLayerConfiguration();
            conf.confs = this.confs;
            conf.hiddenLayerSizes = this.hiddenLayerSizes;
            return conf;

        }


    }


}
