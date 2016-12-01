package org.nd4j.parameterserver.status.play;

import org.junit.Test;
import org.nd4j.parameterserver.model.SubscriberState;

import static junit.framework.TestCase.assertEquals;

/**
 * Created by agibsonccc on 12/1/16.
 */
public class StorageTests {

    @Test
    public void testStorage() {
        StatusStorage statusStorage = new InMemoryStatusStorage();
        StatusStorage mapDb = new MapDbStatusStorage();
        assertEquals(SubscriberState.empty(),statusStorage.getState(-1));
        assertEquals(SubscriberState.empty(),mapDb.getState(-1));


        SubscriberState noEmpty = SubscriberState.builder().isMaster(true)
                .serverState("master").streamId(1).build();
        statusStorage.updateState(noEmpty);
        mapDb.updateState(noEmpty);
        assertEquals(noEmpty,statusStorage.getState(1));
        assertEquals(noEmpty,mapDb.getState(1));
    }

}
