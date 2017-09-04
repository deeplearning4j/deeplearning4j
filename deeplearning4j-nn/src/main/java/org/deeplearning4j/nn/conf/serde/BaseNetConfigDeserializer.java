package org.deeplearning4j.nn.conf.serde;

import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.BaseLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.nd4j.linalg.learning.config.*;
import org.nd4j.linalg.schedule.ISchedule;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;
import org.nd4j.shade.jackson.core.JsonParser;
import org.nd4j.shade.jackson.core.JsonProcessingException;
import org.nd4j.shade.jackson.databind.DeserializationContext;
import org.nd4j.shade.jackson.databind.JsonDeserializer;
import org.nd4j.shade.jackson.databind.JsonMappingException;
import org.nd4j.shade.jackson.databind.deser.ResolvableDeserializer;
import org.nd4j.shade.jackson.databind.deser.std.StdDeserializer;

import java.io.IOException;
import java.util.Map;

/**
 * A custom (abstract) deserializer that handles backward compatibility (currently only for updater refactoring that
 * happened after 0.8.0). This is used for both MultiLayerConfiguration and ComputationGraphConfiguration.<br>
 * We deserialize the config using the default deserializer, then handle the new IUpdater (which will be null for
 * 0.8.0 and earlier configs) if necessary
 *
 * Overall design: http://stackoverflow.com/questions/18313323/how-do-i-call-the-default-deserializer-from-a-custom-deserializer-in-jackson
 *
 * @author Alex Black
 */
public abstract class BaseNetConfigDeserializer<T> extends StdDeserializer<T> implements ResolvableDeserializer {

    protected final JsonDeserializer<?> defaultDeserializer;

    public BaseNetConfigDeserializer(JsonDeserializer<?> defaultDeserializer, Class<T> deserializedType) {
        super(deserializedType);
        this.defaultDeserializer = defaultDeserializer;
    }

    @Override
    public abstract T deserialize(JsonParser jp, DeserializationContext ctxt)
                    throws IOException, JsonProcessingException;


    protected void handleUpdaterBackwardCompatibility(Layer[] layers) {
        //Updater configuration changed after 0.8.0 release
        //Previously: enumerations and a bunch of fields. Now: classes
        //Here, we manually create the appropriate Updater instances, if the iupdater field is empty
        for (int i = 0; i < layers.length; i++) {
            Layer l = layers[i];
            if (l == null || !(l instanceof BaseLayer) || ((BaseLayer) l).getIUpdater() != null) {
                //OK - no need to manually handle IUpdater instances for this layer
                continue;
            }

            BaseLayer bl = (BaseLayer) l;

            Updater u = bl.getUpdater();
            double lr = Double.NaN; //TODO bl.getLearningRate();
            double eps = bl.getEpsilon();
            double rho = bl.getRho();

            switch (u) {
                case SGD:
                    bl.setIUpdater(new Sgd(lr));
                    break;
                case ADAM:
                    double meanDecay = bl.getAdamMeanDecay();
                    double varDecay = bl.getAdamVarDecay();
                    bl.setIUpdater(Adam.builder().learningRate(lr).beta1(meanDecay).beta2(varDecay).epsilon(eps)
                                    .build());
                    break;
                case ADADELTA:
                    bl.setIUpdater(new AdaDelta(rho, eps));
                    break;
                case NESTEROVS:
                    Map<Integer, Double> momentumSchedule = bl.getMomentumSchedule();
                    double momentum = bl.getMomentum();
                    if(momentumSchedule != null){
                        ISchedule ms = new MapSchedule(ScheduleType.ITERATION, momentumSchedule);
                        bl.setIUpdater(new Nesterovs(lr, ms));
                    } else {
                        bl.setIUpdater(new Nesterovs(lr, momentum));
                    }
                    break;
                case ADAGRAD:
                    bl.setIUpdater(new AdaGrad(lr, eps));
                    break;
                case RMSPROP:
                    double rmsDecay = bl.getRmsDecay();
                    bl.setIUpdater(new RmsProp(lr, rmsDecay, eps));
                    break;
                case NONE:
                    bl.setIUpdater(new NoOp());
                    break;
                case CUSTOM:
                    //No op - shouldn't happen
                    break;
            }
        }
    }

    @Override
    public void resolve(DeserializationContext ctxt) throws JsonMappingException {
        ((ResolvableDeserializer) defaultDeserializer).resolve(ctxt);
    }
}
