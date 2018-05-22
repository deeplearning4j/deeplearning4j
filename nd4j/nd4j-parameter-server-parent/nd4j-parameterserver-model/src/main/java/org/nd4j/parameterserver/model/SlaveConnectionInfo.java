package org.nd4j.parameterserver.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serializable;

/**
 * Slave connection info,
 * including the connection url,
 * and the associated master.
 *
 * @author Adam Gibson
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class SlaveConnectionInfo implements Serializable {
    private String connectionUrl;
    private String masterUrl;
}
