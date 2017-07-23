package org.nd4j;

import com.esotericsoftware.kryo.Kryo;
import de.javakaffee.kryoserializers.SynchronizedCollectionsSerializer;
import de.javakaffee.kryoserializers.UnmodifiableCollectionsSerializer;
import org.apache.spark.serializer.KryoRegistrator;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Spark KryoRegistrator for using Nd4j with Spark + Kryo
 * Use via:
 * sparkConf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer");
 * sparkConf.set("spark.kryo.registrator", "org.nd4j.Nd4jRegistrator");
 *
 * @author Alex Black
 */
public class Nd4jRegistrator implements KryoRegistrator {
    @Override
    public void registerClasses(Kryo kryo) {
        kryo.register(Nd4j.getBackend().getNDArrayClass(), new Nd4jSerializer());
        kryo.register(Nd4j.getBackend().getComplexNDArrayClass(), new Nd4jSerializer());

        //Also register Java types (synchronized/unmodifiable collections), which will fail by default
        UnmodifiableCollectionsSerializer.registerSerializers(kryo);
        SynchronizedCollectionsSerializer.registerSerializers(kryo);
    }
}
