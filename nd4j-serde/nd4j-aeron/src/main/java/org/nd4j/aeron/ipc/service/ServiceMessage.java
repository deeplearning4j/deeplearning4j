package org.nd4j.aeron.ipc.service;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.extern.slf4j.Slf4j;

import java.io.Serializable;

/**
 * @author raver119@gmail.com
 */
@Data
@Builder
@Slf4j
@NoArgsConstructor
@AllArgsConstructor
public class ServiceMessage implements Serializable {
    public enum MessageType {
        NEGOTIATE,
        ELECTION,
        REROLL,
        ASSIGN,
    }

    private MessageType messageType;

    private int payloadA;
    private int payloadB;
}
