//
// @author raver119@gmail.com
//

#ifndef LIBND4J_SHAPELIST_H
#define LIBND4J_SHAPELIST_H

#include <vector>
#include <shape.h>

namespace nd4j {
    class ND4J_EXPORT ShapeList {
    protected:
        std::vector<int*> _shapes;

        bool _autoremovable = false;
        bool _workspace = false;
    public:
        ShapeList(int* shape = nullptr);
        ShapeList(std::initializer_list<int*> shapes);
        ShapeList(std::initializer_list<int*> shapes, bool isWorkspace);
        ShapeList(std::vector<int*>& shapes);
        //ShapeList(bool autoRemovable);

        ~ShapeList();

        std::vector<int*>* asVector();
        void destroy();
        int size();
        int* at(int idx);
        void push_back(int *shape);
        void push_back(std::vector<int>& shape);

        /**
         * PLEASE NOTE: This method should be called ONLY if shapes were generated at workspaces. Otherwise you'll get memory leak
         */
        void detach();
    };
}


#endif //LIBND4J_SHAPELIST_H
