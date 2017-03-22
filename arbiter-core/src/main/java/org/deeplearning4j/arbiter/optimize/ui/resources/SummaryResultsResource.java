/*-
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
package org.deeplearning4j.arbiter.optimize.ui.resources;

import org.deeplearning4j.arbiter.optimize.runner.CandidateStatus;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.ws.rs.*;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/** Results for the results table (non-expanded)*/
@Path("/results")
@Produces(MediaType.APPLICATION_JSON)
public class SummaryResultsResource {

    public static final Logger log = LoggerFactory.getLogger(SummaryResultsResource.class);
    private List<CandidateStatus> statusList = new ArrayList<>();

    @GET
    public Response getCandidateStatus(){
        log.trace("GET for candidate status with current status: {}",statusList);

        return Response.ok(statusList).build();
    }

    @POST
    @Path("/update")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response update(List<CandidateStatus> statusList){
        log.trace("POST for results with new status: {}",statusList);
        this.statusList = statusList;
        return Response.ok(Collections.singletonMap("status","ok")).build();
    }

}
