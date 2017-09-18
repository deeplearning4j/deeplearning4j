//
// @author raver119@gmail.com
//

#ifndef LIBND4J_SHAPELIST_H
#define LIBND4J_SHAPELIST_H

#include <vector>
#include <shape.h>

namespace nd4j {
    class ShapeList {
    protected:
        std::vector<int*> _shapes;
    public:
        ShapeList(int* shape = nullptr) {
            if (shape != nullptr)
                _shapes.push_back(shape);
        }

        ShapeList(std::initializer_list<int*> shapes) {
            for (auto v:shapes)
                _shapes.push_back(v);
        }

        ShapeList(std::vector<int*>& shapes) {
            for (auto v:shapes)
                _shapes.push_back(v);
        }

        ~ShapeList() {
            //
        }

        std::vector<int*>* asVector() {
            return &_shapes;
        }

        void destroy() {
            for (auto v:_shapes)
                delete[] v;
        }

        int size() {
            return (int) _shapes.size();
        }

        int* at(int idx) {
            return _shapes.at(idx);
        }

        void push_back(int *shape) {
            _shapes.push_back(shape);
        }

        void push_back(std::vector<int>& shape) {
            int dLen = shape::shapeInfoLength(shape.at(0));

            if (shape.size() != dLen)
                throw "Bad shape was passed in";

            auto nShape = new int[dLen];
            std::memcpy(nShape, shape.data(), shape::shapeInfoByteLength(shape.at(0)));

            _shapes.push_back(nShape);
        }
    };
}


#endif //LIBND4J_SHAPELIST_H
