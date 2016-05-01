package org.deeplearning4j.spark.util;

import akka.actor.ActorSystem;
import akka.serialization.Serialization;
import akka.serialization.SerializationExtension;
import akka.serialization.Serializer;

/**
 *
 * Test serialization
 *
 * @author Adam Gibson
 */
public class SerializationTester {


    /**
     * Testing akka serialization
     * @param system the system to test on
     * @param test the object to test
     */
    public static void testSerialization(ActorSystem system,Object test) throws Exception {
        // Get the Serialization Extension
        Serialization serialization = SerializationExtension.get(system);
        // Find the Serializer for it
        Serializer serializer = serialization.findSerializerFor(test);
        serializer.toBinary(test);
        serializer.fromBinary(serializer.toBinary(test));


    }

    /**
     * Test serialization
     * @param test the object to test
     */
    public static void testSerialization(Object test) {
        ActorSystem as = ActorSystem.create("testserde");
        try {
            testSerialization(as, test);
        } catch (Exception e) {
            e.printStackTrace();
        }
        finally {
            as.shutdown();
        }
    }



}
