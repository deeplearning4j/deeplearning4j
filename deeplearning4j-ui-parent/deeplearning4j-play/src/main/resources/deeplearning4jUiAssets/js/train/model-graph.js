function renderModelGraph(){
    $.ajax({
        url: "/train/model/graph",
        async: true,
        error: function (query, status, error) {
            console.log("Error getting data: " + error);
        },
        success: function (data) {
            console.log("Keys: " + Object.keys(data));

            createGraph(data);
        }
    });
}

function createGraph(data){

    //Generate the elements data
    var vertexNames = data["vertexNames"];    //List<String>
    var vertexTypes = data["vertexTypes"];    //List<String>
    var vertexInputs = data["vertexInputs"];  //int[][]
    var vertexInfos = data["vertexInfo"];     //List<Map<String,String>>
    var vertexCount = vertexNames.length;

    //Layer Styles
    var layerStyles = {
      "Activation": ["#800000", "rectangle"],
      "AutoEncoder": ["#2874A6","rectangle"],
      "BaseOutput": ["#2ECC71","rectangle"],
      "BasePretrainNetwork": ["#9A7D0A","rectangle"],
      "BaseRecurrent": ["#212F3D","rectangle"],
      "BatchNormalization": ["#008080","rectangle"],
      "Convolution": ["#424949","rectangle"],
      "Dense": ["#000080","rectangle"],
      "Embedding": ["#CD5C5C","rectangle"],
      "FeedForward": ["#9B59B6","rectangle"],
      "GravesBidirectionalLSTM": ["#0B5345","rectangle"],
      "GravesLSTM": ["#2b0b06","rectangle"],
      "Input": ["#008000","vee"],
      "InputTypeUtil": ["#DFB391","rectangle"],
      "LocalResponseNormalization": ["#F1948A","rectangle"],
      "Output": ["#FF0000","ellipse"],
      "RBM": ["#F5A45D","rectangle"],
      "RnnOutput": ["#7D3C98","rectangle"],
      "Subsampling": ["#7FB3D5","rectangle"],
      "L2Vertex": ["#800000","triangle"],
      "LayerVertex": ["#2874A6","triangle"],
      "MergeVertex": ["#2ECC71","triangle"],
      "PreprocessorVertex": ["#9A7D0A","triangle"],
      "StackVertex": ["#212F3D","triangle"],
      "SubsetVertex": ["#008080","triangle"],
      "UnstackVertex": ["#424949","triangle"],
      "DuplicateToTimeSeriesVertex": ["#000080","triangle"],
      "LastTimeStepVertex": ["#CD5C5C","triangle"]
    };

    var nodes = [];
    var edges = [];
    for(var i=0; i<vertexNames.length; i++ ){

        //Find correct layer color and shape
        if (Object.keys(layerStyles).indexOf(vertexTypes[i]) > 0 ) {
          layerColor = layerStyles[vertexTypes[i]][0];
          layerShape = layerStyles[vertexTypes[i]][1];
        } else {
          layerColor = "#000000";
          layerShape = "octagon";
        }

        var obj = {
            id: i,
            name: vertexTypes[i] + '\n(' + vertexNames[i] +')',
            faveColor: layerColor,
            faveShape: layerShape,
            onclick: "renderLayerTable()"
        };
        nodes.push({ data: obj} );

        //Edges:
        var inputsToCurrent = vertexInputs[i];
        for(var j=0; j<inputsToCurrent.length; j++ ){
            var e = {
                source: inputsToCurrent[j],
                target: i,
                faveColor: '#A9A9A9',
                strength: 100
            };
            edges.push({ data: e} );
        }
    }

    var elementsToRender = {
        nodes: nodes,
        edges: edges
    };


    $('#layers').cytoscape({
        layout: {
            name: 'dagre',
            padding: 10
        },

        style: cytoscape.stylesheet()
            .selector('node')
            .css({
                'shape': 'data(faveShape)',
                'width': '100',
                'height': '50',
                'content': 'data(name)',
                'text-valign': 'center',
                'text-outline-width': 2,
                'text-outline-color': 'data(faveColor)',
                'background-color': 'data(faveColor)',
                'color': '#fff',
                'text-wrap': 'wrap',
                'font-size': '17px'
            })
            .selector(':selected')
            .css({
                'border-width': 3,
                'border-color': '#333'
            })
            .selector('edge')
            .css({
                'curve-style': 'bezier',
                'opacity': 0.666,
                'width': 'mapData(strength, 70, 100, 2, 6)',
                'target-arrow-shape': 'triangle',
                'source-arrow-shape': 'circle',
                'line-color': 'data(faveColor)',
                'source-arrow-color': 'data(faveColor)',
                'target-arrow-color': 'data(faveColor)'
            })
            .selector('edge.questionable')
            .css({
                'line-style': 'dotted',
                'target-arrow-shape': 'diamond'
            })
            .selector('.faded')
            .css({
                'opacity': 0.25,
                'text-opacity': 0
            }),

        elements: elementsToRender,

        ready: function () {
            window.cy = this;
            if (vertexCount <= 4) {
                cy.zoom(1.6);
                cy.center();
            } else if (vertexCount > 4 && vertexCount <=6) {
                cy.zoom(1);
                cy.center();
            }
            else {
                cy.zoom(1);
                cy.panBy({x: -50, y:0});
            }
            cy.panningEnabled(true);
            cy.autoungrabify(true);
            cy.zoomingEnabled(false);
        }
    });

    cy.on('click', 'node', function (evt) {
        setSelectedVertex(this.id());
    });

}