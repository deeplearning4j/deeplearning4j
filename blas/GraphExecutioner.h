//
// @author raver119@gmail.com
//

#ifndef LIBND4J_GRAPHEXECUTIONER_H
#define LIBND4J_GRAPHEXECUTIONER_H

#include <graph/generated/node_generated.h>
#include <graph/generated/graph_generated.h>

#include <graph/Variable.h>
#include <graph/VariableSpace.h>
#include <graph/Node.h>
#include <graph/Graph.h>
#include <sys/stat.h>

#define TF_INPUT "Placeholder"
#define TF_CONST "Const"
#define TF_VAR "VariableV2"

namespace nd4j {
    namespace graph {

        long getFileSize(const char * filename);

        uint8_t* readFlatBuffers(const char * filename);

        template <typename T>
        class GraphExecutioner {
        protected:


        public:
            //static Nd4jStatus executeFlatNode(nd4j::graph::Graph *graph, nd4j::graph::Node *node, nd4j::graph::VariableSpace<float> *variableSpace);

            /**
             * This method executes given Graph
             * @return
             */
            static Nd4jStatus execute(nd4j::graph::Graph<T> *graph);


            /**
             * This method executes graph stored at given FlatBuffers pointer
             *
             * @param pointer Pointer to FlatBuffer
             * @return pointer to FlatBuffer with result
             */
            static Nd4jPointer executeFlatBuffer(Nd4jPointer pointer);


            static Graph<T> *importFromTensorFlow(const char *fileName);


            static Graph<T> *importFromFlatBuffers(const char *filename);
        };
    }
}

long nd4j::graph::getFileSize(const char * filename) {
    struct stat stat_buf;
    int rc = stat(filename, &stat_buf);
    return rc == 0 ? stat_buf.st_size : -1;
}

uint8_t* nd4j::graph::readFlatBuffers(const char * filename) {
    long fileLen = nd4j::graph::getFileSize(filename);
    uint8_t * data = new uint8_t[fileLen];

    nd4j_verbose("File length: %i\n", fileLen);

    FILE *in = fopen(filename, "rb");
    int cnt = 0;

    while (cnt < fileLen) {
        fread(data + cnt, 1, 1, in);

        cnt++;
    }
    fclose(in);

    return data;
}

template <typename T>
Graph<T>* nd4j::graph::GraphExecutioner<T>::importFromFlatBuffers(const char *filename) {
    uint8_t* data = nd4j::graph::readFlatBuffers(filename);

    auto fg = GetFlatGraph(data);
    auto restoredGraph = new Graph<float>(fg);

    delete[] data;

    return restoredGraph;
}

#endif //LIBND4J_GRAPHEXECUTIONER_H
