/*
 *
 *  * Copyright 2015 Skymind,Inc.
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

package org.deeplearning4j.scaleout.statetracker.hazelcast;


import io.dropwizard.Application;
import io.dropwizard.setup.Bootstrap;
import io.dropwizard.setup.Environment;
import org.deeplearning4j.scaleout.api.statetracker.StateTracker;
import org.deeplearning4j.util.SerializationUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import javax.ws.rs.GET;
import javax.ws.rs.POST;
import javax.ws.rs.PUT;
import javax.ws.rs.Path;
import javax.ws.rs.PathParam;
import javax.ws.rs.Produces;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;

/**
 * Hazelcast resource
 * @author Adam Gibson
 */
@Path("/statetracker")
@Produces(MediaType.APPLICATION_JSON)
public class StateTrackerDropWizardResource extends Application<HazelCastConf> implements Serializable {

    private static final Logger log = LoggerFactory.getLogger(StateTrackerDropWizardResource.class);

    private StateTracker stateTracker;
    private Environment env;
    public StateTrackerDropWizardResource(StateTracker stateTracker) {
        this.stateTracker = stateTracker;
    }

    @Override
    public void initialize(Bootstrap<HazelCastConf> hazelCastConfBootstrap) {
    }

    @Override
    public void run(HazelCastConf hazelCastConf, Environment environment) throws Exception {
        environment.jersey().register(this);
        this.env = environment;
    }


    @GET
    @Path("/minibatch")
    public Response currMiniBatchSize() {
        return Response.ok(Collections.singletonMap("minibatch", stateTracker.miniBatchSize())).build();
    }

    @GET
    @Path("/numbatchessofar")
    public Response numBatches() {
        return Response.ok(Collections.singletonMap("numbatches",stateTracker.numBatchesRan())).build();
    }

    @POST
    @Path("/minibatch/{num}")
    public Response setMiniBatchSizeRest(@PathParam("num") int num) {
        stateTracker.setMiniBatchSize(num);
        return Response.ok(Collections.singletonMap("status","mini batch to " + num)).build();
    }

    @GET
    @Path("/jobs")
    public Response jobs() {
        return Response.ok(stateTracker.jobIds()).build();
    }

    @GET
    @Path("/phase")
    public Response currentState() {
        return Response.ok(Collections.singletonMap("phase","finetune")).build();
    }

    @GET
    @Path("/workers")
    public Response listWorkers() {
        return Response.ok(new ArrayList<>(stateTracker.workers())).build();
    }

    @GET
    @Path("/printmodel")
    public Response printModel() throws Exception {
        return Response.ok(Collections.singletonMap("model",stateTracker.getCurrent().toString())).build();
    }



    @GET
    @Path("/workers/num")
    public Response listWorkersSize() {
        return Response.ok(new ArrayList<>(stateTracker.workers()).size()).build();
    }

    @GET
    @Path("/updates/num")
    public Response listUpdatesSize() {
        return Response.ok(stateTracker.workerUpdates().size()).build();
    }


    @GET
    @Path("/model.ser")
    @Produces(MediaType.APPLICATION_OCTET_STREAM)
    public Response getFile() throws Exception {
        Serializable u =  stateTracker.getCurrent();
        File file = new File("savedmodel.ser");
        SerializationUtils.saveObject(u, file);
        return Response.ok(file, MediaType.APPLICATION_OCTET_STREAM)
                .header("Content-Disposition", "attachment; filename=\"" + file.getName() + "\"" ) //optional
                .build();
    }

    @PUT
    @Path("/save")
    public Response saveModel() {
        log.info("Saving model...");
        try {
            Serializable u = stateTracker.getCurrent();
            SerializationUtils.saveObject(u,new File("savedmodel.ser"));

        }catch(Exception e) {
            return Response.ok(Collections.singletonMap("status", e.getMessage())).build();

        }
        return Response.ok(Collections.singletonMap("status", "saved")).build();
    }



    public void shutdown() {
        if(env != null && env.getAdminContext() != null)
            env.getAdminContext().shutdown();

    }



}
