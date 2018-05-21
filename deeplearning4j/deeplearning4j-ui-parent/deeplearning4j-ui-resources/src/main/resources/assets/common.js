function buildSessionSelector(event) {
  $.ajax({
                              url:"/sessions?event=" + event,
                              async: true,
                              error: function (query, status, error) {
                                  /*$.notify({
                                      title: '<strong>No connection!</strong>',
                                      message: 'DeepLearning4j UiServer seems to be down!'
                                  },{
                                      type: 'danger',
                                      placement: {
                                          from: "top",
                                          align: "center"
                                          },
                                  });
                                  setTimeout(buildSessionSelector(event), 2000);
                                  */
                              },
                              success: function( data ) {
                                 if (data == undefined || data.length == 0) {
                                    $("#sessionSelector").append("<option value='0' selected>No sessions available</option>");
                                    return;
                                 }



                                 for (var i = 0; i < data.length; i++ ) {
                                    $("#sessionSelector").append("<option value='"+ data[0]+"'>ID: "+data[0]+"</option>");
                                 }
                              }
     });
}

    function getParameterByName(name) {
        url = window.location.href;
        name = name.replace(/[\[\]]/g, "\\$&");
        var regex = new RegExp("[?&]" + name + "(=([^&#]*)|&|#|$)"),
            results = regex.exec(url);
        if (!results) return null;
        if (!results[2]) return '';
        return decodeURIComponent(results[2].replace(/\+/g, " "));
    }