package org.nd4j.parameterserver.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serializable;

/**
 * Status of a master node, covered
 * both by the master node itself and its responder.
 *
 * @author Adam Gibson
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class MasterStatus implements Serializable {
    private String master, responder;
    private int responderN;


    /**
     * Returns true if bth
     * the master and responder are started.
     * @return
     */
    public boolean started() {
        return master.equals(ServerState.STARTED.name().toLowerCase())
                        && responder.equals(ServerState.STARTED.name().toLowerCase());
    }

}
