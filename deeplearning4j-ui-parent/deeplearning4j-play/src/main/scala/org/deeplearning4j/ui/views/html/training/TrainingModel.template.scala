
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
                        """),_display_(/*42.26*/i18n/*42.30*/.getMessage("train.session.label")),format.raw/*42.64*/("""
                        """),format.raw/*43.25*/("""<select id="sessionSelect" onchange='selectNewSession()'>
                            <option>(Session ID)</option>
                        </select>
                    </div>
                    <div id="workerSelectDiv" style="display:none; float:right;">
                        """),_display_(/*48.26*/i18n/*48.30*/.getMessage("train.session.worker.label")),format.raw/*48.71*/("""
                        """),format.raw/*49.25*/("""<select id="workerSelect" onchange='selectNewWorker()'>
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
                                """),_display_(/*71.34*/i18n/*71.38*/.getMessage("train.nav.language")),format.raw/*71.71*/("""</span></a>
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
                    renderModelPage(true);
                """),format.raw/*287.17*/("""}"""),format.raw/*287.18*/(""");
        </script>

            <!-- Execute periodically (every 2 sec) -->
        <script>
                setInterval(function () """),format.raw/*292.41*/("""{"""),format.raw/*292.42*/("""
                    """),format.raw/*293.21*/("""renderModelPage(false);
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
                  DATE: Sun Nov 06 21:19:42 AEDT 2016
                  SOURCE: C:/DL4J/Git/deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-play/src/main/views/org/deeplearning4j/ui/views/training/TrainingModel.scala.html
                  HASH: 33ddbc164d57cf4aaa4aa23dbdf3cb4fe3485954
                  MATRIX: 598->1|731->39|759->41|882->138|894->142|944->172|2729->1930|2742->1934|2793->1964|2941->2085|2954->2089|3009->2123|3063->2149|3379->2438|3392->2442|3454->2483|3508->2509|4237->3210|4251->3214|4306->3247|4485->3398|4499->3402|4551->3432|4706->3559|4720->3563|4773->3594|4921->3714|4935->3718|4991->3752|5250->3984|5263->3988|5317->4021|7601->6276|7631->6277|7682->6299|7867->6455|7897->6456|7944->6474|8471->6973|8485->6977|8554->7024|9014->7456|9028->7460|9102->7512|9193->7711|9257->7881|9320->7945|9387->8019|9459->8208|9531->8398|9603->8590|9670->8636|9733->8680|9796->8714|10043->8932|10058->8936|10125->8980|10295->9121|10310->9125|10377->9169|10549->9312|10564->9316|10632->9361|11045->9746|11059->9750|11139->9807|11296->9936|11310->9940|11378->9986|11449->10028|11731->10282|11745->10286|11816->10335|12101->10592|12115->10596|12192->10650|12259->10688|12369->10770|12383->10774|12451->10820|12522->10862|12801->11113|12815->11117|12877->11157|13163->11415|13177->11419|13245->11464|13312->11502|13423->11585|13437->11589|13505->11635|13576->11677|13856->11929|13870->11933|13939->11980|14563->12576|14577->12580|14647->12628|19896->17848|19926->17849|19977->17871|20087->17952|20117->17953|20286->18093|20316->18094|20367->18116|20437->18157|20467->18158
                  LINES: 20->1|25->1|26->2|31->7|31->7|31->7|64->40|64->40|64->40|66->42|66->42|66->42|67->43|72->48|72->48|72->48|73->49|89->65|89->65|89->65|90->66|90->66|90->66|91->67|91->67|91->67|92->68|92->68|92->68|95->71|95->71|95->71|127->103|127->103|128->104|132->108|132->108|133->109|146->122|146->122|146->122|155->131|155->131|155->131|156->132|157->133|158->134|159->135|160->136|161->137|162->138|163->139|164->140|165->141|166->142|166->142|166->142|167->143|167->143|167->143|168->144|168->144|168->144|174->150|174->150|174->150|176->152|176->152|176->152|177->153|183->159|183->159|183->159|187->163|187->163|187->163|188->164|189->165|189->165|189->165|190->166|196->172|196->172|196->172|200->176|200->176|200->176|201->177|202->178|202->178|202->178|203->179|209->185|209->185|209->185|220->196|220->196|220->196|308->284|308->284|309->285|311->287|311->287|316->292|316->292|317->293|318->294|318->294
                  -- GENERATED --
              */
          