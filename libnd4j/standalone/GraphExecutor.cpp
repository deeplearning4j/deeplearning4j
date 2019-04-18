/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// @author yves@iv-devs.com
//

#include "LabelsImagenet.h"
#include <graph/Variable.h>
#include <graph/VariableSpace.h>
#include <NDArray.h>
#include <NDArrayFactory.h>
#include <GraphExecutioner.h>
#include <graph/profiling/GraphProfilingHelper.h>
#include <iostream>
#include <vector>
#include <map>
#include <stdio.h>
#include <png.h>
// Header downloaded from a third party during cmake configuration. Run cmake or buildnativeoperations.sh once
// in order to download it.
#include "thirdparty/klib/ketopt.h"

using std::cout;
using std::cerr;
using std::endl;
using std::string;

using namespace nd4j;

enum preProcessingType {NONE=0, INCEPTION=1, VGG=2};

void abort_(const char * s, ...)
{
        va_list args;
        va_start(args, s);
        vfprintf(stderr, s, args);
        fprintf(stderr, "\n");
        va_end(args);
        abort();
}

void read_size(const char *file_name, int& width, int& height){
    png_structp png_ptr;
    png_infop info_ptr;

    unsigned char header[8];

    FILE *fp = fopen(file_name, "rb");
    if (!fp)
            abort_("[read_png_file] File %s could not be opened for reading", file_name);
    fread(header, 1, 8, fp);
    if (png_sig_cmp(header, 0, 8))
            abort_("[read_png_file] File %s is not recognized as a PNG file", file_name);


    png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

    if (!png_ptr)
            abort_("[read_png_file] png_create_read_struct failed");

    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr)
            abort_("[read_png_file] png_create_info_struct failed");

    if (setjmp(png_jmpbuf(png_ptr)))
            abort_("[read_png_file] Error during init_io");

    png_init_io(png_ptr, fp);
    png_set_sig_bytes(png_ptr, 8);

    png_read_info(png_ptr, info_ptr);

    width = png_get_image_width(png_ptr, info_ptr);
    height = png_get_image_height(png_ptr, info_ptr);
}

void read_png_file(const char *file_name, NDArray* output) {

    int x, y;

    int width, height;
    png_byte color_type;
    png_byte bit_depth;

    png_structp png_ptr;
    png_infop info_ptr;
    int number_of_passes;
    png_bytep * row_pointers;

    unsigned char header[8];    // 8 is the maximum size that can be checked

    FILE *fp = fopen(file_name, "rb");
    if (!fp)
            abort_("[read_png_file] File %s could not be opened for reading", file_name);
    fread(header, 1, 8, fp);
    if (png_sig_cmp(header, 0, 8))
            abort_("[read_png_file] File %s is not recognized as a PNG file", file_name);


    png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

    if (!png_ptr)
            abort_("[read_png_file] png_create_read_struct failed");

    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr)
            abort_("[read_png_file] png_create_info_struct failed");

    if (setjmp(png_jmpbuf(png_ptr)))
            abort_("[read_png_file] Error during init_io");

    png_init_io(png_ptr, fp);
    png_set_sig_bytes(png_ptr, 8);

    png_read_info(png_ptr, info_ptr);

    width = png_get_image_width(png_ptr, info_ptr);
    height = png_get_image_height(png_ptr, info_ptr);
    color_type = png_get_color_type(png_ptr, info_ptr);
    bit_depth = png_get_bit_depth(png_ptr, info_ptr);

    number_of_passes = png_set_interlace_handling(png_ptr);
    png_read_update_info(png_ptr, info_ptr);


    if (setjmp(png_jmpbuf(png_ptr)))
            abort_("[read_png_file] Error during read_image");

    row_pointers = (png_bytep*) malloc(sizeof(png_bytep) * height);
    for (y=0; y<height; y++)
            row_pointers[y] = (png_byte*) malloc(png_get_rowbytes(png_ptr,info_ptr));

    png_read_image(png_ptr, row_pointers);
    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
            unsigned char r,g,b;
            r = ((unsigned char *)(row_pointers[y]))[x*3];
            g = ((unsigned char *)(row_pointers[y]))[x*3+1];
            b = ((unsigned char *)(row_pointers[y]))[x*3+2];
            output->p<float>(0,y,x,0, (float)(r));
            output->p<float>(0,y,x,1, (float)(g));
            output->p<float>(0,y,x,2, (float)(b));
            /*if(y==50){
                printf("%d %d %d | ",r,g,b);
            }*/
        }
    }
    //printf("\n");
    fclose(fp);
}


void read_csv(string filename, NDArray* output, int width){
    char * line;
    FILE *fp;
    size_t size=256;
    fp = fopen(filename.c_str(),"r");
    if(!fp) {
        printf("Could not open file %s\n", filename.c_str());
        exit(-1);
    }
    int ind=0;
    line = (char*)(malloc(256));
    ssize_t read=1;
    read=getline(&line, &size, fp);
    while(read>0){
        int x = (ind/3) % width;
        int y = ind/(3*width);
        output->p<float>(0,y,x, ind%3, atof(line));
        read=getline(&line, &size, fp);
        ind++;
    }
    free(line);
    return;
}

void showVariables(Graph* graph){
    std::vector<Variable*> variablesVector = graph->getVariableSpace()->getVariables();
    printf("Variables: \n");
    for(int i=0;i<variablesVector.size();i++){
        Variable* var = variablesVector[i];
        printf(" * %d %s", var->id(), var->getName()->c_str());
        if(var->shape().size()>0){
            printf("\t variable shape:");
            bool skip_first=true;
            for(auto s:var->shape()){
                if(!skip_first) printf("%lld ", s);
                skip_first=false;
            }
        }
        if(var->hasNDArray()){
            printf("\t array shape:");
            long long * shapeInfo = var->getNDArray()->shapeInfo();
            int rank = shape::rank(shapeInfo);
            int lim = shape::shapeInfoLength(rank);
            for (int i = 0; i < lim; i++) {
                printf("%lld", (long long) shapeInfo[i]);

                if (i < lim - 1) {
                    printf(", ");
                }
            }
        }
        printf("\n");
    }
    printf("\n");
}

void showPlaceholders(Graph* graph){
    std::vector<Variable*> variablesVector = *(graph->getVariableSpace()->getPlaceholders());
    printf("Placeholders: \n");
    for(int i=0;i<variablesVector.size();i++){
        Variable* var = variablesVector[i];
        printf(" * %s\n", var->getName()->c_str());
    }
    printf("\n");
}

void rgbToBgrConversion(NDArray* img){
    int width = img->getShapeAsVector()[1];
    int height = img->getShapeAsVector()[2];

    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
            float r,g,b;
            r = img->e<float>(0,y,x,0);
            g = img->e<float>(0,y,x,1);
            b = img->e<float>(0,y,x,2);
            img->p<float>(0,y,x,0, b);
            img->p<float>(0,y,x,1, g);
            img->p<float>(0,y,x,2, r);
        }
    }
}

void vggPreprocessing(NDArray* img){

    const float VGG_MEAN_R = 0.485f;
    const float VGG_MEAN_G = 0.456f;
    const float VGG_MEAN_B = 0.406f;

    const float VGG_STD_R = 0.229f;
    const float VGG_STD_G = 0.224f;
    const float VGG_STD_B = 0.225f;

    int width = img->getShapeAsVector()[1];
    int height = img->getShapeAsVector()[2];

    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
            float r,g,b;
            r = ((img->e<float>(0,y,x,0)/255.0f) - VGG_MEAN_R);///VGG_STD_R;
            g = ((img->e<float>(0,y,x,1)/255.0f) - VGG_MEAN_G);///VGG_STD_G;
            b = ((img->e<float>(0,y,x,2)/255.0f) - VGG_MEAN_B);///VGG_STD_B;
            img->p<float>(0,y,x,0, r);
            img->p<float>(0,y,x,1, g);
            img->p<float>(0,y,x,2, b);
        }
    }
}

void inceptionPreprocessing(NDArray* img){
    int width = img->getShapeAsVector()[1];
    int height = img->getShapeAsVector()[2];

    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
            float r,g,b;
            r = img->e<float>(0,y,x,0);
            g = img->e<float>(0,y,x,1);
            b = img->e<float>(0,y,x,2);
            img->p<float>(0,y,x,0, (((float)(r))/128.0f) - 1.0f);
            img->p<float>(0,y,x,1, (((float)(g))/128.0f) - 1.0f);
            img->p<float>(0,y,x,2, (((float)(b))/128.0f) - 1.0f);
        }
    }
}

string guessInputLayer(Graph* graph, int width, int height){
    std::vector<Variable*> variablesVector = graph->getVariableSpace()->getVariables();
    std::vector<std::pair<int, string> > guessScore;
    for(int i=0;i<variablesVector.size();i++){
        Variable* var = variablesVector[i];
        std::pair<int, string> guess(0, *var->getName());
        if(var->shape().size()>0){
            guess.first++;
            bool skip_first=true;
            for(auto s:var->shape()){
                if(s==width || s==height)
                    guess.first++;
                skip_first=false;
            }
        }
        guessScore.push_back(guess);
    }
    std::sort(guessScore.begin(), guessScore.end());
    std::reverse(guessScore.begin(), guessScore.end());
    if(guessScore.size()>0)
        return guessScore[0].second;
    else
        return "";
}

string guessOutputLayer(Graph* graph){
    std::vector<std::pair<int, string> > guessScore;
    std::map<int, int> connections;
    for(int nodeIndex:(*graph->nodes())){
        Node* node = graph->nodeById(nodeIndex);
        for (int e = 0; e < node->input()->size(); e++) {
            auto in = node->input()->at(e);
            connections[in.first]++;
        }
    }
    for(int nodeIndex:(*graph->nodes())){
        Node* node = graph->nodeById(nodeIndex);
        std::pair<int, string> guess(0, *node->getName());
        //printf("%s %d\n", node->name()->c_str(), connections[node->id()]);
        if(connections[node->id()]==0){
            //printf("%s gained a point\n", node->name()->c_str());
            guess.first++;
        }
        Variable* var = graph->getVariableSpace()->getVariable(node->id());
        if(var->hasNDArray()){
            long long * shapeInfo = var->getNDArray()->shapeInfo();
            int rank = shape::rank(shapeInfo);
            //printf(" %s ", node->name()->c_str());
            for (int i = 1; i <= rank; i++) {
                //printf(" %d ", shapeInfo[i]);
            }
            //printf("\n");
            for (int i = 1; i <= rank; i++) {
                if(shapeInfo[i]==1000 || shapeInfo[i]==1001){
                    //printf("%s gained a point\n", node->name()->c_str());
                    guess.first++;
                }
                if(shapeInfo[i]!=1000 && shapeInfo[i]!=1001 && shapeInfo[i]!=1){
                    //printf("%s lost a point\n", node->name()->c_str());
                    guess.first--;
                }
            }
        }
        guessScore.push_back(guess);
    }

    std::sort(guessScore.begin(), guessScore.end());
    std::reverse(guessScore.begin(), guessScore.end());
    for(auto g:guessScore){
        //printf("%d %s\n", g.first, g.second.c_str());
    }
    if(guessScore.size()>0)
        return guessScore[0].second;
    else
        return "";
}

void printUsage(){
    cout << "Usage: " <<endl;
    cout << "./GraphExecutor model_file.fb [image_file.png] [ARGS]" << endl;
    cout << endl;
    cout << "If no image file is present, the graph will not be executed." <<endl;
    cout << "Available arguments: " << endl;
    cout << "\t-i input layer name (if not specified the program will try to guess it)" << endl;
    cout << "\t-o output layer name (if not specified the program will try to guess it)" << endl;
    cout << "\t--pre preprocessing to apply to the input:" << endl;
    cout << "\t\t0: None" << endl;
    cout << "\t\t1: INCEPTION (default)" << endl;
    cout << "\t\t2: VGG" << endl;
    cout << "\t--rgb2bgr switches the color channels" << endl;
    cout << "\t--graph displays the graph of the model" << endl;
    cout << "\t--variables lists the variables of the model" << endl;
    cout << "\t--placeholders lists the placeholders of the model" << endl;
    cout << "\t-h this help message" << endl;

    /*
     *         printf("Usage: ./GraphExecutor model.fb image.png [input_layer_name] [output_layer_name] [preprocessing] [rgb_to_bgr]\n\n"
               "preprocessing:\n\t0: None\n\t1:INCEPTION [default]\n\t2:VGG\n\n"
               "rgb_to_bgr\n\t0:false [default]\n\t1:true\n");
*/
}


int main(int argc, char** argv){

    /** Options:
     * -d Debug
     * -v Verbose
     * --graph display graph
     * --variables show variables
     * --placeholders
     * --profile
     * -h help
     * -i input layer
     * -o output layer
     **/


    nd4j::Environment::getInstance()->setDebug(false);
    nd4j::Environment::getInstance()->setVerbose(false);
    bool displayGraph = false;
    bool displayVariables = false;
    bool displayPlaceholders = false;
    bool profiling = false;
    enum preProcessingType preProcessing = INCEPTION;
    bool rgbToBgr = false;
    bool verbose = false;
    string inputLayerName = "";
    string outputLayerName = "";

    static ko_longopt_t longopts[] = {
        { "verbose", ko_no_argument,         302 },
        { "graph",   ko_no_argument,         303 },
        { "variables", ko_no_argument,       304 },
        { "placeholders",   ko_no_argument,  305 },
        { "profile",   ko_no_argument,       306 },
        { "help",      ko_no_argument,       307 },
        { "pre",       ko_required_argument, 308 },
        { "rgb2bgr",   ko_no_argument,       309 },
        { NULL, 0, 0 }
    };
    ketopt_t opt = KETOPT_INIT;
    int i, c;
    while ((c = ketopt(&opt, argc, argv, 1, "vhi:o:", longopts)) >= 0) {
        switch(c){
        case 'v':
        case 302:
            nd4j::Environment::getInstance()->setDebug(true);
            nd4j::Environment::getInstance()->setVerbose(true);
            break;
        case 303:
            displayGraph = true;
            break;
        case 304:
            displayVariables = true;
            break;
        case 305:
            displayPlaceholders= true;
            break;
        case 306:
            profiling = true;
            break;
        case 'i':
            if(!opt.arg){
                cerr << "Error: -i requires an argument" << endl;
                printUsage();
                return -1;
            }
            else
                inputLayerName.assign(opt.arg);
            break;
        case 'o':
            if(!opt.arg){
                cerr << "Error: -o requires an argument" << endl;
                printUsage();
                return -1;
            }
            else
                outputLayerName.assign(opt.arg);
            break;
        case 307:
        case 'h':
            printUsage();
            return 0;
            break;
        case 308:
            if(!opt.arg){
                cerr << "Error: -pre requires an argument" << endl;
                printUsage();
                return -1;
            }
            else
                preProcessing = (preProcessingType)(atoi(opt.arg));
            break;
        case 309:
            rgbToBgr = true;
            break;
        case '?':
        default:
            cerr << "Unkwnown argument " << (char)(opt.opt?opt.opt:' ') << endl;
            printUsage();
            exit(-1);
        }
    }
    string modelFilename = "";
    string imageFilename = "";

    int argcount=0;
    for (i = opt.ind; i < argc; ++i){
      if(argcount==0) modelFilename.assign(argv[i]);
      if(argcount==1) imageFilename.assign(argv[i]);
      argcount++;
    }

    auto graph = GraphExecutioner::importFromFlatBuffers(modelFilename.c_str());
    graph->buildGraph();

    if(!imageFilename.empty()){
        int width, height;
        read_size(imageFilename.c_str(), width, height);

        NDArray* inputArray = NDArrayFactory::create_<float>('c', {1, height, width, 3});
        read_png_file(imageFilename.c_str(), inputArray);
        //inputArray->assign(1.0f);
        if(rgbToBgr){
            rgbToBgrConversion(inputArray);
        }
        if(preProcessing == VGG){
            vggPreprocessing(inputArray);
        }
        else if (preProcessing == INCEPTION){
            inceptionPreprocessing(inputArray);
        }

        if(inputLayerName.empty()){
            inputLayerName = guessInputLayer(graph, width, height);
            printf("Guessed the input layer as \"%s\"\n", inputLayerName.c_str());
        }

        Variable* input = new Variable(inputArray, inputLayerName.c_str());
        graph->getVariableSpace()->replaceVariable(input);

        //read_csv("/home/yves/dl4j/datasets/mobilev1.csv", &inputArray, width);

        if(profiling){
            Environment::getInstance()->setProfiling(true);
            auto profile = GraphProfilingHelper::profile(graph, 1, 0);
            profile->printOut();
        }
        GraphExecutioner::execute(graph);

        if(outputLayerName.empty()){
            outputLayerName = guessOutputLayer(graph);
            printf("Guessed the output layer as \"%s\"\n", outputLayerName.c_str());
        }


        NDArray* result = graph->getVariableSpace()->getVariable(&outputLayerName)->getNDArray();
        std::vector<float> rvec = result->getBufferAsVector<float>();
        LabelsImagenet imageNetLabels;
        std::map<int, std::string>& labels = imageNetLabels.labels;

        for(int k=0;k<20;k++){
            printf("(%d): %f ", (int)(result->argMax()), rvec[result->argMax()]);
            printf("\t %s\n", labels[result->argMax()-1].c_str());
            result->p<float>(result->argMax(), 0);
        }
    }
    if(displayGraph)
        graph->printOut();
    if(displayPlaceholders)
        showPlaceholders(graph);
    if(displayVariables)
        showVariables(graph);



}
