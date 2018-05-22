package org.nd4j.parameterserver.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serializable;
import java.util.List;

/**
 * Created by agibsonccc on 10/9/16.
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class MasterConnectionInfo implements Serializable {
    private String connectionUrl;
    private String responderUrl;
    private List<String> slaveUrls;
}
