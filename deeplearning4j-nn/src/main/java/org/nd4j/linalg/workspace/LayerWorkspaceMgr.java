package org.nd4j.linalg.workspace;

import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.ndarray.INDArray;

public class LayerWorkspaceMgr extends BaseWorkspaceMgr<NetArrayType> {
    
    public LayerWorkspaceMgr(){
        INDArray arr = null;

    }

    public static Builder builder(){

    }

    public static class Builder {

        private LayerWorkspaceMgr mgr;

        public Builder(){
            mgr = new LayerWorkspaceMgr();
        }

        public Builder defaultDummyWorkspace(){
            for(NetArrayType t : NetArrayType.values()){
                if(!mgr.configMap.containsKey(t)){
                    mgr.setDummyWorkspace(t);
                }
            }
            return this;
        }

        public Builder withDummy(NetArrayType type){
            mgr.setDummyWorkspace(type);
            return this;
        }

        public Builder defaultWorkspace(String workspaceName, WorkspaceConfiguration configuration){
            for(NetArrayType t : NetArrayType.values()){
                if(!mgr.configMap.containsKey(t)){
                    with(t, workspaceName, configuration);
                }
            }
            return this;
        }

        public Builder with(NetArrayType type, String workspaceName, WorkspaceConfiguration configuration){
            mgr.setConfiguration(type, configuration);
            mgr.setWorkspaceName(type, workspaceName);
            return this;
        }

        public LayerWorkspaceMgr build(){
            return mgr;
        }

    }
    
}
