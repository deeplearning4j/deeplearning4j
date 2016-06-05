/*
 *
 *  * Copyright 2016 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */
package org.deeplearning4j.arbiter.optimize.runner.listener.candidate;

import org.deeplearning4j.arbiter.optimize.runner.Status;
import org.deeplearning4j.arbiter.optimize.ui.ClientProvider;
import org.deeplearning4j.ui.api.Component;
import org.deeplearning4j.ui.api.LengthUnit;
import org.deeplearning4j.ui.components.component.ComponentDiv;
import org.deeplearning4j.ui.components.component.style.StyleDiv;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.ws.rs.client.Entity;
import javax.ws.rs.client.WebTarget;
import javax.ws.rs.core.MediaType;

public class UICandidateStatusListenerImpl implements UICandidateStatusListener {

    private static final StyleDiv styleDiv = new StyleDiv.Builder()
            .width(100, LengthUnit.Percent)
            .height(100, LengthUnit.Percent)
            .build();

    private static final Logger log = LoggerFactory.getLogger(UICandidateStatusListener.class);
    private final int candidateNumber;
    private WebTarget target;

    public UICandidateStatusListenerImpl(int candidateNumber){
        this.candidateNumber = candidateNumber;
        //TODO don't hardcode
        target = ClientProvider.getClient().target("http://localhost:8080/modelResults/update/" + candidateNumber);
    }


    @Override
    public void reportStatus(Status status, Component... uiElements) {
        target.request(MediaType.APPLICATION_JSON).accept(MediaType.APPLICATION_JSON)
                .post(Entity.entity(
                        new ComponentDiv(styleDiv, uiElements), MediaType.APPLICATION_JSON));
        log.info("Update posted for candidate {}",candidateNumber);
    }
}
