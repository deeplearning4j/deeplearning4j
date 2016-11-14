function onNavClick(source, errorMsg) {
    console.log("Clicked: " + source);
    var reqURL = "train/" + source;
    $.ajax({
        url: reqURL,
        success: function (data) {
            $("#overviewNavDiv").removeClass("navElementSelected");
            $("#modelNavDiv").removeClass("navElementSelected");
            $("#systemNavDiv").removeClass("navElementSelected");
            $("#helpNavDiv").removeClass("navElementSelected");
            $("#" + source + "NavDiv").addClass("navElementSelected");
            $("#mainContentDiv").html(data);
        },
        error: function (query, status, error) {
            $.notify(errorMsg, "error");
        }
    });
}
function changeLanguage(errorMsg) {
    var currSelected = $("#navLangSelect").val();
    $.ajax({
        url: "setlang/" + currSelected,
        success: function () {
            location.reload(true);
            console.log("Selected language: " + currSelected);
        },
        error: function (query, status, error) {
            $.notify(errorMsg, "error");
        }
    });
}
function setSessionIDDivContents() {
    $.ajax({
        url: "/train/sessions/current",
        success: function (data) {
            $("#navSessionIDValue").html(data);
        }
    });
}
function setLanguageSelectorValue() {
    $.ajax({
        url: "/lang/getCurrent",
        success: function (data) {
            $("#navLangSelect").val(data);
        }
    });
}
function getAndProcessUpdate() {
    console.log("Updating");
    if ($("#homeNavDiv").hasClass("navElementSelected")) {
        getAndProcessUpdateOverview();
    }
    else if ($("#modelNavDiv").hasClass("navElementSelected")) {
    }
    else if ($("#systemNavDiv").hasClass("navElementSelected")) {
    }
    else if ($("#helpNavDiv").hasClass("navElementSelected")) {
    }
}
function getAndProcessUpdateOverview() {
    $.ajax({
        url: "train/overview/data",
        success: function (data) {
            console.log("Overview data: " + data);
        },
        error: function (query, status, error) {
            $.notify("Error", "error");
        }
    });
}
//# sourceMappingURL=dl4j-play-ui.js.map