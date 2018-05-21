function renderModelGraph(){
    $.ajax({
        url: "/train/model/graph",
        async: true,
        error: function (query, status, error) {
            console.log("Error getting data: " + error);
        },
        success: function (data) {
            createGraph(data);
        }
    });
}

function createGraph(data){

    //Generate the elements data
    var vertexNames = data["vertexNames"];    //List<String>
    if (typeof vertexNames == 'undefined') return;  //No data
    var vertexTypes = data["vertexTypes"];    //List<String>
    var vertexInputs = data["vertexInputs"];  //int[][]
    var vertexInfos = data["vertexInfo"];     //List<Map<String,String>>
    var vertexCount = vertexNames.length;

    //Layer Styles
    var layerStyles = {
      "Activation": ["#CD6155", "rectangle"],
      "AutoEncoder": ["#641E16","rectangle"],
      "BaseOutput": ["#AF7AC5","rectangle"],
      "BasePretrainNetwork": ["#512E5F","rectangle"],
      "BaseRecurrent": ["#5499C7","rectangle"],
      "BatchNormalization": ["#154360","rectangle"],
      "Convolution": ["#1B2631","rectangle"],
      "Dense": ["#EB984E","rectangle"],
      "Embedding": ["#F4D03F","rectangle"],
      "FeedForward": ["#7D6608","rectangle"],
      "GravesBidirectionalLSTM": ["#1ABC9C","rectangle"],
      "GravesLSTM": ["#6E2C00","rectangle"],
      "Input": ["#145A32","vee"],
      "InputTypeUtil": ["#5D6D7E","rectangle"],
      "LocalResponseNormalization": ["#52BE80","rectangle"],
      "Output": ["#922B21","ellipse"],
      "RBM": ["#48C9B0","rectangle"],
      "RnnOutput": ["#0E6251","rectangle"],
      "Subsampling": ["#4D5656","rectangle"],
      "L2Vertex": ["#78281F","triangle"],
      "LayerVertex": ["#4A235A","triangle"],
      "MergeVertex": ["#1B4F72","triangle"],
      "PreprocessorVertex": ["#0B5345","triangle"],
      "StackVertex": ["#186A3B","triangle"],
      "SubsetVertex": ["#7E5109","triangle"],
      "UnstackVertex": ["#6E2C00","triangle"],
      "DuplicateToTimeSeriesVertex": ["#424949","triangle"],
      "LastTimeStepVertex": ["#17202A","triangle"]
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
//            if (vertexCount <= 4) {
//                cy.zoom(1.6);
//                cy.center();
//            } else if (vertexCount > 4 && vertexCount <=6) {
//                cy.zoom(1);
//                cy.center();
//            }
//            else {
//                cy.zoom(1);
//                cy.panBy({x: -50, y:0});
//            }
            cy.panningEnabled(true);
            cy.autoungrabify(true);
            cy.zoomingEnabled(true);
            cy.fit(elementsToRender, 50);
        }
    });

    cy.on('click', 'node', function (evt) {
        setSelectedVertex(this.id());
    });

}