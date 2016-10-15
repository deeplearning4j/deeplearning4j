/// <reference path="../../typedefs/jquery.d.ts" />
/// <reference path="../../typedefs/notify.d.ts" />

function onNavClick(source: String, errorMsg: String) {

    console.log("Clicked: " + source);

    var reqURL: string = "train/" + source;

    $.ajax({
        url: reqURL,
        success: function (data) {
            $("#mainContentDiv").html(data);
        },
        error: function (query, status, error) {
            $.notify(errorMsg,"error");
        }
    });
}