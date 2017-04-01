package org.nd4j.parameterserver.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serializable;

/**
 * Created by agibsonccc on 10/9/16.
 */
@Builder
@Data
@NoArgsConstructor
@AllArgsConstructor
public class ServerTypeJson implements Serializable {
    private String type;
}
