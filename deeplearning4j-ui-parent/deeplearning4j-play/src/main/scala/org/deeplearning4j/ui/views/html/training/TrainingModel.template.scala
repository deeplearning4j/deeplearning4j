
package org.deeplearning4j.ui.views.html.training

import play.twirl.api._
import play.twirl.api.TemplateMagic._


     object TrainingModel_Scope0 {
import models._
import controllers._
import play.api.i18n._
import views.html._
import play.api.templates.PlayMagic._
import play.api.mvc._
import play.api.data._

class TrainingModel extends BaseScalaTemplate[play.twirl.api.HtmlFormat.Appendable,Format[play.twirl.api.HtmlFormat.Appendable]](play.twirl.api.HtmlFormat) with play.twirl.api.Template1[org.deeplearning4j.ui.api.I18N,play.twirl.api.HtmlFormat.Appendable] {

  /**/
  def apply/*1.2*/(i18n: org.deeplearning4j.ui.api.I18N):play.twirl.api.HtmlFormat.Appendable = {
    _display_ {
      {


Seq[Any](format.raw/*1.40*/("""
"""),format.raw/*2.1*/("""<!DOCTYPE html>
<html lang="en">
    <head>

        <meta charset="utf-8">
        <title>"""),_display_(/*7.17*/i18n/*7.21*/.getMessage("train.pagetitle")),format.raw/*7.51*/("""</title>
            <!-- start: Mobile Specific -->
        <meta name="viewport" content="width=device-width, initial-scale=1">
            <!-- end: Mobile Specific -->

        <link id="bootstrap-style" href="/assets/css/bootstrap.min.css" rel="stylesheet">
        <link href="/assets/css/bootstrap-responsive.min.css" rel="stylesheet">
        <link id="base-style" href="/assets/css/style.css" rel="stylesheet">
        <link id="base-style-responsive" href="/assets/css/style-responsive.css" rel="stylesheet">
        <link href='http://fonts.googleapis.com/css?family=Open+Sans:300italic,400italic,600italic,700italic,800italic,400,300,600,700,800&subset=latin,cyrillic-ext,latin-ext' rel='stylesheet' type='text/css'>
        <link rel="shortcut icon" href="/assets/img/favicon.ico">

            <!-- The HTML5 shim, for IE6-8 support of HTML5 elements -->
            <!--[if lt IE 9]>
	  	<script src="http://html5shim.googlecode.com/svn/trunk/html5.js"></script>
		<link id="ie-style" href="/assets/css/ie.css" rel="stylesheet"/>
	<![endif]-->

            <!--[if IE 9]>
		<link id="ie9style" href="/assets/css/ie9.css" rel="stylesheet"/>
	<![endif]-->
    </head>

    <body>
            <!-- Start Header -->
        <div class="navbar">
            <div class="navbar-inner">
                <div class="container-fluid">
                    <a class="btn btn-navbar" data-toggle="collapse" data-target=".top-nav.nav-collapse,.sidebar-nav.nav-collapse">
                        <span class="icon-bar"></span>
                        <span class="icon-bar"></span>
                        <span class="icon-bar"></span>
                    </a>
                    <a class="brand" href="index.html"><span>"""),_display_(/*40.63*/i18n/*40.67*/.getMessage("train.pagetitle")),format.raw/*40.97*/("""</span></a>
                    <div id="sessionSelectDiv" style="display:none; float:right">
                        Session:
                        <select id="sessionSelect" onchange='selectNewSession()'>
                            <option>(Session ID)</option>
                        </select>
                    </div>
                    <div id="workerSelectDiv" style="display:none; float:right;">
                        Worker:
                        <select id="workerSelect" onchange='selectNewWorker()'>
                            <option>(Worker ID)</option>
                        </select>
                    </div>
                </div>
            </div>
        </div>
            <!-- End Header -->

        <div class="container-fluid-full">
            <div class="row-fluid">

                    <!-- Start Main Menu -->
                <div id="sidebar-left" class="span2">
                    <div class="nav-collapse sidebar-nav">
                        <ul class="nav nav-tabs nav-stacked main-menu">
                            <li><a href="overview"><i class="icon-bar-chart"></i> <span class="hidden-tablet">"""),_display_(/*65.112*/i18n/*65.116*/.getMessage("train.nav.overview")),format.raw/*65.149*/("""</span></a></li>
                            <li class="active"><a href="javascript:void(0);"><i class="icon-tasks"></i> <span class="hidden-tablet">"""),_display_(/*66.134*/i18n/*66.138*/.getMessage("train.nav.model")),format.raw/*66.168*/("""</span></a></li>
                            <li><a href="system"><i class="icon-dashboard"></i> <span class="hidden-tablet">"""),_display_(/*67.110*/i18n/*67.114*/.getMessage("train.nav.system")),format.raw/*67.145*/("""</span></a></li>
                            <li><a href="help"><i class="icon-star"></i> <span class="hidden-tablet">"""),_display_(/*68.103*/i18n/*68.107*/.getMessage("train.nav.userguide")),format.raw/*68.141*/("""</span></a></li>
                            <li>
                                <a class="dropmenu" href="javascript:void(0);"><i class="icon-folder-close-alt"></i> <span class="hidden-tablet">
                                    Language</span></a>
                                <ul>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('en', 'model')"><i class="icon-file-alt"></i> <span class="hidden-tablet">
                                        English</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('ja', 'model')"><i class="icon-file-alt"></i> <span class="hidden-tablet">
                                        日本語</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('zh', 'model')"><i class="icon-file-alt"></i> <span class="hidden-tablet">
                                        中文</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('ko', 'model')"><i class="icon-file-alt"></i> <span class="hidden-tablet">
                                        한글</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('ru', 'model')"><i class="icon-file-alt"></i> <span class="hidden-tablet">
                                        русский</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('uk', 'model')"><i class="icon-file-alt"></i> <span class="hidden-tablet">
                                        український</span></a></li>
                                </ul>
                            </li>
                        </ul>
                    </div>
                </div>
                    <!-- End Main Menu -->

                <noscript>
                    <div class="alert alert-block span10">
                        <h4 class="alert-heading">Warning!</h4>
                        <p>You need to have <a href="http://en.wikipedia.org/wiki/JavaScript" target="_blank">
                            JavaScript</a>
                            enabled to use this site.</p>
                    </div>
                </noscript>

                <style>
                /* Graph */
                #layers """),format.raw/*103.25*/("""{"""),format.raw/*103.26*/("""
                    """),format.raw/*104.21*/("""height: 725px; /* IE8 */
                    height: 90vh;
                    width: 100%;
                    border: 2px solid #eee;
                """),format.raw/*108.17*/("""}"""),format.raw/*108.18*/("""
                """),format.raw/*109.17*/("""</style>

                    <!-- Start Content -->
                <div id="content" class="span10">

                    <div class="row-fluid span6">
                        <div id="layers"></div>
                    </div>
                        <!-- Start Layer Details -->
                    <div class="row-fluid span6" id="layerDetails">

                        <div class="box">
                            <div class="box-header">
                                <h2><b>"""),_display_(/*122.41*/i18n/*122.45*/.getMessage("train.model.layerInfoTable.title")),format.raw/*122.92*/("""</b></h2>
                            </div>
                            <div class="box-content">
                                <table class="table table-bordered table-striped table-condensed" id="layerInfo"></table>
                            </div>
                        </div>

                        <div class="box">
                            <div class="box-header">
                                <h2><b>"""),_display_(/*131.41*/i18n/*131.45*/.getMessage("train.overview.chart.updateRatioTitle")),format.raw/*131.97*/(""": log<sub>10</sub></b></h2>
                                """),format.raw/*132.171*/("""
                                """),format.raw/*133.169*/("""
                                """),format.raw/*134.63*/("""
                                    """),format.raw/*135.73*/("""
                                        """),format.raw/*136.188*/("""
                                        """),format.raw/*137.189*/("""
                                        """),format.raw/*138.191*/("""
                                    """),format.raw/*139.45*/("""
                                """),format.raw/*140.43*/("""
                                """),format.raw/*141.33*/("""<ul class="nav tab-menu nav-tabs" style="position:relative; top: -11px; right: -12px;">
                                    <li id="mmRatioTab"><a href="javascript:void(0);" onclick="setSelectMeanMagChart('ratios')">"""),_display_(/*142.130*/i18n/*142.134*/.getMessage("train.model.meanmag.btn.ratio")),format.raw/*142.178*/("""</a></li>
                                    <li id="mmParamTab"><a href="javascript:void(0);" onclick="setSelectMeanMagChart('paramMM')">"""),_display_(/*143.131*/i18n/*143.135*/.getMessage("train.model.meanmag.btn.param")),format.raw/*143.179*/("""</a></li>
                                    <li id="mmUpdateTab"><a href="javascript:void(0);" onclick="setSelectMeanMagChart('updateMM')">"""),_display_(/*144.133*/i18n/*144.137*/.getMessage("train.model.meanmag.btn.update")),format.raw/*144.182*/("""</a></li>
                                </ul>
                            </div>
                            <div class="box-content">
                                <div id="meanmag" class="center" style="height: 300px;" ></div>
                                <p id="hoverdata"><span id="updateRatioTitleSmallLog10"><b>log<sub>
                                    10</sub> """),_display_(/*150.47*/i18n/*150.51*/.getMessage("train.overview.chart.updateRatioTitleShort")),format.raw/*150.108*/("""</b></span> <span id="yMeanMagnitudes">
                                    0</span>
                                    , <b>"""),_display_(/*152.43*/i18n/*152.47*/.getMessage("train.overview.charts.iteration")),format.raw/*152.93*/("""
                                        """),format.raw/*153.41*/(""":</b> <span id="xMeanMagnitudes">0</span></p>
                            </div>
                        </div>

                        <div class="box">
                            <div class="box-header">
                                <h2><b>"""),_display_(/*159.41*/i18n/*159.45*/.getMessage("train.model.activationsChart.title")),format.raw/*159.94*/("""</b></h2>
                            </div>
                            <div class="box-content">
                                <div id="activations" class="center" style="height: 300px;" ></div>
                                <p id="hoverdata"><b>"""),_display_(/*163.55*/i18n/*163.59*/.getMessage("train.model.activationsChart.titleShort")),format.raw/*163.113*/("""
                                    """),format.raw/*164.37*/(""":</b> <span id="yActivations">0</span>
                                    , <b>"""),_display_(/*165.43*/i18n/*165.47*/.getMessage("train.overview.charts.iteration")),format.raw/*165.93*/("""
                                        """),format.raw/*166.41*/(""":</b> <span id="xActivations">0</span></p>
                            </div>
                        </div>

                        <div class="box">
                            <div class="box-header">
                                <h2><b>"""),_display_(/*172.41*/i18n/*172.45*/.getMessage("train.model.lrChart.title")),format.raw/*172.85*/("""</b></h2>
                            </div>
                            <div class="box-content">
                                <div id="learningrate" class="center" style="height: 300px;" ></div>
                                <p id="hoverdata"><b>"""),_display_(/*176.55*/i18n/*176.59*/.getMessage("train.model.lrChart.titleShort")),format.raw/*176.104*/("""
                                    """),format.raw/*177.37*/(""":</b> <span id="yLearningRate">0</span>
                                    , <b>"""),_display_(/*178.43*/i18n/*178.47*/.getMessage("train.overview.charts.iteration")),format.raw/*178.93*/("""
                                        """),format.raw/*179.41*/(""":</b> <span id="xLearningRate">0</span></p>
                            </div>
                        </div>

                        <div class="box">
                            <div class="box-header">
                                <h2><b>"""),_display_(/*185.41*/i18n/*185.45*/.getMessage("train.model.paramHistChart.title")),format.raw/*185.92*/("""</b></h2>
                                <div id="paramhistSelected" style="float: left"></div>
                                <div id="paramHistButtonsDiv" style="float: right"></div>
                            </div>
                            <div class="box-content">
                                <div id="parametershistogram" class="center" style="height: 300px;"></div>
                            </div>
                        </div>

                        <div class="box">
                            <div class="box-header">
                                <h2><b>"""),_display_(/*196.41*/i18n/*196.45*/.getMessage("train.model.updateHistChart.title")),format.raw/*196.93*/("""</b></h2>
                                <div id="updatehistSelected" style="float: left"></div>
                                <div id="updateHistButtonsDiv" style="float: right"></div>
                            </div>
                            <div class="box-content">
                                <div id="updateshistogram" class="center" style="height: 300px;"></div>
                            </div>
                        </div>

                    </div>
                        <!-- End Layer Details-->

                        <!-- Begin Zero State -->
                    <div class="row-fluid span6" id="zeroState">
                        <div class="box">
                            <div class="box-header">
                                <h2><b>Getting Started</b></h2>
                            </div>
                            <div class="box-content">
                                <div class="page-header">
                                    <h1>Layer Visualization UI</h1>
                                </div>
                                <div class="row-fluid">
                                    <div class="span12">
                                        <h2>Overview</h2>
                                        <p>
                                            The layer visualization UI renders network structure dynamically. Users can inspect node layer parameters by clicking on the various elements of the GUI to see general information as well as overall network information such as performance.
                                        </p>
                                        <h2>Actions</h2>
                                        <p>On the <b>left</b>, you will find an interactive layer visualization.</p>
                                        <p>
                                    <ul>
                                        <li><b>Clicking</b> - Click on a layer to load network performance metrics.</li>
                                        <li><b>Scrolling</b>
                                            - Drag the GUI with your mouse or touchpad to move the model around. </li>
                                    </ul>
                                        </p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                        <!-- End Zero State-->
                </div>
                    <!-- End Content -->
            </div> <!-- End Container -->
        </div> <!-- End Row Fluid-->

        <!-- Start JavaScript-->
        <script src="/assets/js/jquery-1.9.1.min.js"></script>
        <script src="/assets/js/jquery-migrate-1.0.0.min.js"></script>
        <script src="/assets/js/jquery-ui-1.10.0.custom.min.js"></script>
        <script src="/assets/js/jquery.ui.touch-punch.js"></script>
        <script src="/assets/js/modernizr.js"></script>
        <script src="/assets/js/bootstrap.min.js"></script>
        <script src="/assets/js/jquery.cookie.js"></script>
        <script src="/assets/js/fullcalendar.min.js"></script>
        <script src="/assets/js/jquery.dataTables.min.js"></script>
        <script src="/assets/js/excanvas.js"></script>
        <script src="/assets/js/jquery.flot.js"></script>
        <script src="/assets/js/jquery.flot.pie.js"></script>
        <script src="/assets/js/jquery.flot.stack.js"></script>
        <script src="/assets/js/jquery.flot.resize.min.js"></script>
        <script src="/assets/js/jquery.chosen.min.js"></script>
        <script src="/assets/js/jquery.uniform.min.js"></script>
        <script src="/assets/js/jquery.cleditor.min.js"></script>
        <script src="/assets/js/jquery.noty.js"></script>
        <script src="/assets/js/jquery.elfinder.min.js"></script>
        <script src="/assets/js/jquery.raty.min.js"></script>
        <script src="/assets/js/jquery.iphone.toggle.js"></script>
        <script src="/assets/js/jquery.uploadify-3.1.min.js"></script>
        <script src="/assets/js/jquery.gritter.min.js"></script>
        <script src="/assets/js/jquery.imagesloaded.js"></script>
        <script src="/assets/js/jquery.masonry.min.js"></script>
        <script src="/assets/js/jquery.knob.modified.js"></script>
        <script src="/assets/js/jquery.sparkline.min.js"></script>
        <script src="/assets/js/counter.js"></script>
        <script src="/assets/js/retina.js"></script>
        <script src="/assets/js/custom.js"></script>
        <script src="/assets/js/cytoscape.min.js"></script>
        <script src="/assets/js/dagre.min.js"></script>
        <script src="/assets/js/cytoscape-dagre.js"></script>
        <script src="/assets/js/train/model.js"></script> <!-- Charts and tables are generated here! -->
        <script src="/assets/js/train/model-graph.js"></script> <!-- Layer graph generated here! -->
        <script src="/assets/js/train/train.js"></script> <!-- Common (lang selection, etc) -->

        <!-- Execute once on page load -->
        <script>
                $(document).ready(function () """),format.raw/*284.47*/("""{"""),format.raw/*284.48*/("""
                    """),format.raw/*285.21*/("""renderModelGraph();
                    renderModelPage();
                """),format.raw/*287.17*/("""}"""),format.raw/*287.18*/(""");
        </script>

            <!-- Execute periodically (every 2 sec) -->
        <script>
                setInterval(function () """),format.raw/*292.41*/("""{"""),format.raw/*292.42*/("""
                    """),format.raw/*293.21*/("""renderModelPage();
                """),format.raw/*294.17*/("""}"""),format.raw/*294.18*/(""", 2000);
        </script>
    </body>
</html>
"""))
      }
    }
  }

  def render(i18n:org.deeplearning4j.ui.api.I18N): play.twirl.api.HtmlFormat.Appendable = apply(i18n)

  def f:((org.deeplearning4j.ui.api.I18N) => play.twirl.api.HtmlFormat.Appendable) = (i18n) => apply(i18n)

  def ref: this.type = this

}


}

/**/
object TrainingModel extends TrainingModel_Scope0.TrainingModel
              /*
                  -- GENERATED --
                  DATE: Sun Nov 06 19:08:35 AEDT 2016
                  SOURCE: C:/DL4J/Git/deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-play/src/main/views/org/deeplearning4j/ui/views/training/TrainingModel.scala.html
                  HASH: 74ae211d876ed1d1e4075353438e704842943dbc
                  MATRIX: 598->1|731->39|759->41|882->138|894->142|944->172|2729->1930|2742->1934|2793->1964|3997->3140|4011->3144|4066->3177|4245->3328|4259->3332|4311->3362|4466->3489|4480->3493|4533->3524|4681->3644|4695->3648|4751->3682|7278->6180|7308->6181|7359->6203|7544->6359|7574->6360|7621->6378|8148->6877|8162->6881|8231->6928|8691->7360|8705->7364|8779->7416|8870->7615|8934->7785|8997->7849|9064->7923|9136->8112|9208->8302|9280->8494|9347->8540|9410->8584|9473->8618|9720->8836|9735->8840|9802->8884|9972->9025|9987->9029|10054->9073|10226->9216|10241->9220|10309->9265|10722->9650|10736->9654|10816->9711|10973->9840|10987->9844|11055->9890|11126->9932|11408->10186|11422->10190|11493->10239|11778->10496|11792->10500|11869->10554|11936->10592|12046->10674|12060->10678|12128->10724|12199->10766|12478->11017|12492->11021|12554->11061|12840->11319|12854->11323|12922->11368|12989->11406|13100->11489|13114->11493|13182->11539|13253->11581|13533->11833|13547->11837|13616->11884|14240->12480|14254->12484|14324->12532|19573->17752|19603->17753|19654->17775|19760->17852|19790->17853|19959->17993|19989->17994|20040->18016|20105->18052|20135->18053
                  LINES: 20->1|25->1|26->2|31->7|31->7|31->7|64->40|64->40|64->40|89->65|89->65|89->65|90->66|90->66|90->66|91->67|91->67|91->67|92->68|92->68|92->68|127->103|127->103|128->104|132->108|132->108|133->109|146->122|146->122|146->122|155->131|155->131|155->131|156->132|157->133|158->134|159->135|160->136|161->137|162->138|163->139|164->140|165->141|166->142|166->142|166->142|167->143|167->143|167->143|168->144|168->144|168->144|174->150|174->150|174->150|176->152|176->152|176->152|177->153|183->159|183->159|183->159|187->163|187->163|187->163|188->164|189->165|189->165|189->165|190->166|196->172|196->172|196->172|200->176|200->176|200->176|201->177|202->178|202->178|202->178|203->179|209->185|209->185|209->185|220->196|220->196|220->196|308->284|308->284|309->285|311->287|311->287|316->292|316->292|317->293|318->294|318->294
                  -- GENERATED --
              */
          