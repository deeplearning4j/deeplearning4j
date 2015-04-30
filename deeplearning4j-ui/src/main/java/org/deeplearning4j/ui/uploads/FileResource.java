package org.deeplearning4j.ui.uploads;





import org.glassfish.jersey.media.multipart.FormDataContentDisposition;
import org.glassfish.jersey.media.multipart.FormDataParam;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.ws.rs.*;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;
import java.io.*;

/**
 * Created by Juan on 25/02/2015.
 *
 * Original credit: https://github.com/juanpabloprado/dw-multipart/
 */

public abstract class FileResource {
    private static final Logger LOGGER = LoggerFactory.getLogger(FileResource.class);
    private String filePath = System.getProperty("java.io.tmpdir");

    /**
     * The file path for uploads
     * @param filePath the file path for uploads
     */
    public FileResource(String filePath) {
        this.filePath = filePath;
    }

    @POST
    @Path("/upload")
    @Produces(MediaType.APPLICATION_JSON)
    @Consumes(MediaType.MULTIPART_FORM_DATA)
    public Response uploadFile(
            @FormDataParam("file") InputStream uploadedInputStream,
            @FormDataParam("file") FormDataContentDisposition fileDetail) throws  IOException {
        String uploadedFileLocation = new File(filePath,fileDetail.getFileName()).getAbsolutePath();
        LOGGER.info(uploadedFileLocation);
        // save it
        writeToFile(uploadedInputStream, uploadedFileLocation);
        String output = "File uploaded to : " + uploadedFileLocation;
        return Response.ok(output).build();
    }

    // save uploaded file to new location
    private void writeToFile(InputStream uploadedInputStream, String uploadedFileLocation) throws IOException {
        int read;
        final int BUFFER_LENGTH = 1024;
        final byte[] buffer = new byte[BUFFER_LENGTH];
        OutputStream out = new FileOutputStream(new File(uploadedFileLocation));
        while ((read = uploadedInputStream.read(buffer)) != -1) {
            out.write(buffer, 0, read);
        }
        out.flush();
        out.close();
        handleUpload(new File(uploadedFileLocation));
    }

    public abstract void handleUpload(File path);
}