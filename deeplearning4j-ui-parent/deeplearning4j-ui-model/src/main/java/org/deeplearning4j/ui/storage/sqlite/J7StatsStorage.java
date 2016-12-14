package org.deeplearning4j.ui.storage.sqlite;

import lombok.NonNull;
import org.deeplearning4j.api.storage.Persistable;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.api.storage.StatsStorageListener;
import org.deeplearning4j.api.storage.StorageMetaData;
import org.deeplearning4j.berkeley.Pair;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.sql.*;
import java.util.Collection;
import java.util.List;

/**
 * Created by Alex on 14/12/2016.
 */
public class J7StatsStorage implements StatsStorage {

    private static final String INSERT_META_SQL = "INSERT OR REPLACE INTO StorageMetaData (SessionID, TypeID, ObjectClass, ObjectBytes) VALUES ( ?, ?, ?, ? );";

    private final File file;
    private final Connection connection;

    public J7StatsStorage(@NonNull File file) {
        this.file = file;
        if (!file.exists()) {

        }

        try {
            connection = DriverManager.getConnection("jdbc:sqlite:" + file.getAbsolutePath());
        } catch (Exception e) {
            throw new RuntimeException("Error ninializing J7StatsStorage instance", e);
        }

        try {
            initializeTables();
        } catch (SQLException e) {
            throw new RuntimeException(e);
        }
    }

    private void initializeTables() throws SQLException {

        //Need tables for:
        //(a) Metadata  -> session ID and type ID; class; StorageMetaData as a binary BLOB
        //(b) Static info -> session ID, type ID, worker ID, persistable class, persistable bytes
        //(c) Update info -> session ID, type ID, worker ID, timestamp, update class, update bytes

        //TODO check if tables exist

        Statement statement = connection.createStatement();

        statement.executeUpdate(
                "CREATE TABLE StorageMetaData (" +
                        "SessionID TEXT NOT NULL, " +
                        "TypeID TEXT NOT NULL, " +
                        "ObjectClass TEXT NOT NULL, " +
                        "ObjectBytes BLOB NOT NULL, " +
                        "PRIMARY KEY ( SessionID, TypeID )" +
                        ");");

        statement.executeUpdate(
                "CREATE TABLE StaticInfo (" +
                        "SessionID TEXT NOT NULL, " +
                        "TypeID TEXT NOT NULL, " +
                        "WorkerID TEXT NOT NULL, " +
                        "ObjectClass TEXT NOT NULL, " +
                        "ObjectBytes BLOB NOT NULL, " +
                        "PRIMARY KEY ( SessionID, TypeID, WorkerID )" +
                        ");");

        statement.executeUpdate(
                "CREATE TABLE Updates (" +
                        "SessionID TEXT NOT NULL, " +
                        "TypeID TEXT NOT NULL, " +
                        "WorkerID TEXT NOT NULL, " +
                        "Timestamp INTEGER NOT NULL, " +
                        "ObjectClass TEXT NOT NULL, " +
                        "ObjectBytes BLOB NOT NULL, " +
                        "PRIMARY KEY ( SessionID, TypeID, WorkerID, Timestamp )" +
                        ");");

        statement.close();



    }

    private static Pair<String,byte[]> serializeForDB(Object object){
        String classStr = object.getClass().getName();
        try(ByteArrayOutputStream baos = new ByteArrayOutputStream(); ObjectOutputStream oos = new ObjectOutputStream(baos)){
            oos.writeObject(object);
            oos.close();
            byte[] bytes = baos.toByteArray();
            return new Pair<>(classStr, bytes);
        } catch (IOException e){
            throw new RuntimeException("Error serializing object for storage", e);
        }
    }


    @Override
    public void putStorageMetaData(StorageMetaData storageMetaData) {
        try{
            Pair<String,byte[]> p = serializeForDB(storageMetaData);

            PreparedStatement ps = connection.prepareStatement(INSERT_META_SQL);
            ps.setString(1, storageMetaData.getSessionID());
            ps.setString(2, storageMetaData.getTypeID());
            ps.setString(3, p.getFirst());
            ps.setObject(4, p.getSecond());
            ps.executeUpdate();
        } catch (SQLException e){
            throw new RuntimeException(e);
        }
    }

    @Override
    public void putStorageMetaData(Collection<? extends StorageMetaData> storageMetaData) {

    }

    @Override
    public void putStaticInfo(Persistable staticInfo) {

    }

    @Override
    public void putStaticInfo(Collection<? extends Persistable> staticInfo) {

    }

    @Override
    public void putUpdate(Persistable update) {

    }

    @Override
    public void putUpdate(Collection<? extends Persistable> updates) {

    }

    @Override
    public void close() throws IOException {

    }

    @Override
    public boolean isClosed() {
        return false;
    }

    @Override
    public List<String> listSessionIDs() {
        return null;
    }

    @Override
    public boolean sessionExists(String sessionID) {
        return false;
    }

    @Override
    public Persistable getStaticInfo(String sessionID, String typeID, String workerID) {
        return null;
    }

    @Override
    public List<Persistable> getAllStaticInfos(String sessionID, String typeID) {
        return null;
    }

    @Override
    public List<String> listTypeIDsForSession(String sessionID) {
        return null;
    }

    @Override
    public List<String> listWorkerIDsForSession(String sessionID) {
        return null;
    }

    @Override
    public List<String> listWorkerIDsForSessionAndType(String sessionID, String typeID) {
        return null;
    }

    @Override
    public int getNumUpdateRecordsFor(String sessionID) {
        return 0;
    }

    @Override
    public int getNumUpdateRecordsFor(String sessionID, String typeID, String workerID) {
        return 0;
    }

    @Override
    public Persistable getLatestUpdate(String sessionID, String typeID, String workerID) {
        return null;
    }

    @Override
    public Persistable getUpdate(String sessionID, String typeId, String workerID, long timestamp) {
        return null;
    }

    @Override
    public List<Persistable> getLatestUpdateAllWorkers(String sessionID, String typeID) {
        return null;
    }

    @Override
    public List<Persistable> getAllUpdatesAfter(String sessionID, String typeID, String workerID, long timestamp) {
        return null;
    }

    @Override
    public List<Persistable> getAllUpdatesAfter(String sessionID, String typeID, long timestamp) {
        return null;
    }

    @Override
    public StorageMetaData getStorageMetaData(String sessionID, String typeID) {
        return null;
    }

    @Override
    public void registerStatsStorageListener(StatsStorageListener listener) {

    }

    @Override
    public void deregisterStatsStorageListener(StatsStorageListener listener) {

    }

    @Override
    public void removeAllListeners() {

    }

    @Override
    public List<StatsStorageListener> getListeners() {
        return null;
    }
}
