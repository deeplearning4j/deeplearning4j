
package org.deeplearning4j.ui.views.html.training

import play.twirl.api._
import play.twirl.api.TemplateMagic._


     object TrainingSystem_Scope0 {
import models._
import controllers._
import play.api.i18n._
import views.html._
import play.api.templates.PlayMagic._
import play.api.mvc._
import play.api.data._

class TrainingSystem extends BaseScalaTemplate[play.twirl.api.HtmlFormat.Appendable,Format[play.twirl.api.HtmlFormat.Appendable]](play.twirl.api.HtmlFormat) with play.twirl.api.Template1[org.deeplearning4j.ui.api.I18N,play.twirl.api.HtmlFormat.Appendable] {

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
            <!-- Start Mobile Specific -->
        <meta name="viewport" content="width=device-width, initial-scale=1">
            <!-- End Mobile Specific -->

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
                    <a class="brand" href="index.html"><span>"""),_display_(/*41.63*/i18n/*41.67*/.getMessage("train.pagetitle")),format.raw/*41.97*/("""</span></a>
                    <div id="sessionSelectDiv" style="display:none; float:right">
                        """),_display_(/*43.26*/i18n/*43.30*/.getMessage("train.session.label")),format.raw/*43.64*/("""
                        """),format.raw/*44.25*/("""<select id="sessionSelect" onchange='selectNewSession()'>
                            <option>(Session ID)</option>
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
                            <li><a href="overview"><i class="icon-bar-chart"></i><span class="hidden-tablet"> """),_display_(/*60.112*/i18n/*60.116*/.getMessage("train.nav.overview")),format.raw/*60.149*/("""</span></a></li>
                            <li><a href="model"><i class="icon-tasks"></i><span class="hidden-tablet"> """),_display_(/*61.105*/i18n/*61.109*/.getMessage("train.nav.model")),format.raw/*61.139*/("""</span></a></li>
                            <li class="active"><a href="javascript:void(0);"><i class="icon-dashboard"></i><span class="hidden-tablet"> """),_display_(/*62.138*/i18n/*62.142*/.getMessage("train.nav.system")),format.raw/*62.173*/("""</span></a></li>
                            """),format.raw/*63.161*/("""
                            """),format.raw/*64.29*/("""<li>
                                <a class="dropmenu" href="javascript:void(0);"><i class="icon-folder-close-alt"></i><span class="hidden-tablet">
                                    """),_display_(/*66.38*/i18n/*66.42*/.getMessage("train.nav.language")),format.raw/*66.75*/("""</span></a>
                                <ul>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('en', 'system')"><i class="icon-file-alt"></i> <span class="hidden-tablet"> English</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('ja', 'system')"><i class="icon-file-alt"></i> <span class="hidden-tablet"> 日本語</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('zh', 'system')"><i class="icon-file-alt"></i> <span class="hidden-tablet"> 中文</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('ko', 'system')"><i class="icon-file-alt"></i> <span class="hidden-tablet"> 한글</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('ru', 'system')"><i class="icon-file-alt"></i> <span class="hidden-tablet"> русский</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('uk', 'system')"><i class="icon-file-alt"></i> <span class="hidden-tablet"> український</span></a></li>
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
                            JavaScript</a> enabled to use this site.</p>
                    </div>
                </noscript>

                    <!-- Start Content -->
                <div id="content" class="span10">

                    <div class="row-fluid">

                        <div class="box span12">
                            <div class="box-header">
                                <h2><b>"""),_display_(/*96.41*/i18n/*96.45*/.getMessage("train.system.title")),format.raw/*96.78*/("""</b></h2>
                                <div class="btn-group" style="margin-top: -11px; position:absolute; right: 40px;">
                                <button class="btn dropdown-toggle btn-primary" data-toggle="dropdown">"""),_display_(/*98.105*/i18n/*98.109*/.getMessage("train.system.selectMachine")),format.raw/*98.150*/(""" """),format.raw/*98.151*/("""<span class="caret"></span></button>
                                    <ul class="dropdown-menu" id="systemTab"></ul>
                                </div>
                            </div>
                            <div class="box-content">

                                    <!--Start System Information -->
                                <div class="tab-content">
                                    <div class="tab-pane active">

                                            <!-- System Memory Utilization Chart -->
                                        <div class="row-fluid">

                                            <div class="box span12" id="systemMemoryChart">
                                                <div class="box-header">
                                                    <h2><b>"""),_display_(/*113.61*/i18n/*113.65*/.getMessage("train.system.chart.systemMemoryTitle")),format.raw/*113.116*/(""" """),format.raw/*113.117*/("""%</b></h2>
                                                </div>
                                                <div class="box-content">
                                                    <div id="systemMemoryChartPlot" class="center" style="height: 300px;" ></div>
                                                    <p id="hoverdata"><b>"""),_display_(/*117.75*/i18n/*117.79*/.getMessage("train.system.chart.memoryShort")),format.raw/*117.124*/(""":</b> <span id="y">0</span>, <b>
                                                        """),_display_(/*118.58*/i18n/*118.62*/.getMessage("train.overview.charts.iteration")),format.raw/*118.108*/(""":</b> <span id="x">0</span></p>
                                                </div>
                                            </div>

                                            <!-- GPU Memory Utlization Chart -->
                                            <div class="box span6" id="gpuMemoryChart">
                                                <div class="box-header">
                                                    <h2><b>"""),_display_(/*125.61*/i18n/*125.65*/.getMessage("train.system.chart.gpuMemoryTitle")),format.raw/*125.113*/(""" """),format.raw/*125.114*/("""%</b></h2>
                                                </div>
                                                <div class="box-content">
                                                    <div id="gpuMemoryChartPlot" class="center" style="height: 300px;" ></div>
                                                    <p id="hoverdata"><b>"""),_display_(/*129.75*/i18n/*129.79*/.getMessage("train.system.chart.memoryShort")),format.raw/*129.124*/(""":</b> <span id="y2">0</span>, <b>
                                                        """),_display_(/*130.58*/i18n/*130.62*/.getMessage("train.overview.charts.iteration")),format.raw/*130.108*/(""":</b> <span id="x2">0</span></p>
                                                </div>
                                            </div>

                                        </div>

                                            <!-- Tables -->
                                        <div class="row-fluid">

                                                <!-- Hardware Information -->
                                            <div class="box span12">
                                                <div class="box-header">
                                                    <h2><b>"""),_display_(/*142.61*/i18n/*142.65*/.getMessage("train.system.hwTable.title")),format.raw/*142.106*/("""</b></h2>
                                                </div>
                                                <div class="box-content">
                                                    <table class="table table-striped">
                                                        <thead>
                                                            <tr>
                                                                <th>"""),_display_(/*148.70*/i18n/*148.74*/.getMessage("train.system.hwTable.jvmCurrent")),format.raw/*148.120*/("""</th>
                                                                <th>"""),_display_(/*149.70*/i18n/*149.74*/.getMessage("train.system.hwTable.jvmMax")),format.raw/*149.116*/("""</th>
                                                                <th>"""),_display_(/*150.70*/i18n/*150.74*/.getMessage("train.system.hwTable.offHeapCurrent")),format.raw/*150.124*/("""</th>
                                                                <th>"""),_display_(/*151.70*/i18n/*151.74*/.getMessage("train.system.hwTable.offHeapMax")),format.raw/*151.120*/("""</th>
                                                                <th>"""),_display_(/*152.70*/i18n/*152.74*/.getMessage("train.system.hwTable.jvmProcs")),format.raw/*152.118*/("""</th>
                                                                <th>"""),_display_(/*153.70*/i18n/*153.74*/.getMessage("train.system.hwTable.computeDevices")),format.raw/*153.124*/("""</th>
                                                            </tr>
                                                        </thead>
                                                        <tbody>
                                                            <tr>
                                                                <td id="currentBytesJVM">Loading...</td>
                                                                <td id="maxBytesJVM">Loading...</td>
                                                                <td id="currentBytesOffHeap">Loading...</td>
                                                                <td id="maxBytesOffHeap">Loading...</td>
                                                                <td id="jvmAvailableProcessors">Loading...</td>
                                                                <td id="nComputeDevices">Loading...</td>
                                                            </tr>
                                                        </tbody>
                                                    </table>
                                                </div>
                                            </div>

                                        </div>

                                        <div class="row-fluid">

                                                <!-- Software Information -->
                                            <div class="box span12">
                                                <div class="box-header">
                                                    <h2><b>"""),_display_(/*177.61*/i18n/*177.65*/.getMessage("train.system.swTable.title")),format.raw/*177.106*/("""</b></h2>
                                                </div>
                                                <div class="box-content">
                                                    <table class="table table-striped">
                                                        <thead>
                                                            <tr>
                                                                <th>"""),_display_(/*183.70*/i18n/*183.74*/.getMessage("train.system.swTable.hostname")),format.raw/*183.118*/("""</th>
                                                                <th>"""),_display_(/*184.70*/i18n/*184.74*/.getMessage("train.system.swTable.os")),format.raw/*184.112*/("""</th>
                                                                <th>"""),_display_(/*185.70*/i18n/*185.74*/.getMessage("train.system.swTable.osArch")),format.raw/*185.116*/("""</th>
                                                                <th>"""),_display_(/*186.70*/i18n/*186.74*/.getMessage("train.system.swTable.jvmName")),format.raw/*186.117*/("""</th>
                                                                <th>"""),_display_(/*187.70*/i18n/*187.74*/.getMessage("train.system.swTable.jvmVersion")),format.raw/*187.120*/("""</th>
                                                                <th>"""),_display_(/*188.70*/i18n/*188.74*/.getMessage("train.system.swTable.nd4jBackend")),format.raw/*188.121*/("""</th>
                                                                <th>"""),_display_(/*189.70*/i18n/*189.74*/.getMessage("train.system.swTable.nd4jDataType")),format.raw/*189.122*/("""</th>
                                                            </tr>
                                                        </thead>
                                                        <tbody>
                                                            <tr>
                                                                <td id="hostName">Loading...</td>
                                                                <td id="OS">Loading...</td>
                                                                <td id="OSArchitecture">Loading...</td>
                                                                <td id="jvmName">Loading...</td>
                                                                <td id="jvmVersion">Loading...</td>
                                                                <td id="nd4jBackend">Loading...</td>
                                                                <td id="nd4jDataType">Loading...</td>
                                                            </tr>
                                                        </tbody>
                                                    </table>
                                                </div>
                                            </div>

                                        </div>

                                            """),format.raw/*209.73*/("""
                                        """),format.raw/*210.82*/("""
                                            """),format.raw/*211.73*/("""
                                                """),format.raw/*212.77*/("""
                                                    """),format.raw/*213.88*/("""
                                                """),format.raw/*214.59*/("""
                                                """),format.raw/*215.78*/("""
                                                    """),format.raw/*216.92*/("""
                                                        """),format.raw/*217.68*/("""
                                                            """),format.raw/*218.69*/("""
                                                                """),format.raw/*219.79*/("""
                                                            """),format.raw/*220.70*/("""
                                                        """),format.raw/*221.69*/("""
                                                        """),format.raw/*222.68*/("""
                                                            """),format.raw/*223.69*/("""
                                                                """),format.raw/*224.108*/("""
                                                            """),format.raw/*225.70*/("""
                                                        """),format.raw/*226.69*/("""
                                                    """),format.raw/*227.65*/("""
                                                """),format.raw/*228.59*/("""
                                            """),format.raw/*229.55*/("""
                                        """),format.raw/*230.51*/("""
                                    """),format.raw/*231.37*/("""</div>
                                        <!-- End System Tab -->
                                </div>
                            </div>
                        </div>
                            <!-- End System Tab -->
                    </div><!-- End Row Fluid-->
                </div><!-- End Content -->
            </div><!-- End Container-->
        </div><!-- End Row Fluid-->

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
        <script src="/assets/js/train/system.js"></script> <!-- Charts and tables are generated here! -->
        <script src="/assets/js/train/train.js"></script>   <!-- Common (lang selection, etc) -->

        <!-- Execute once on page load -->
        <script>
                $(document).ready(function () """),format.raw/*277.47*/("""{"""),format.raw/*277.48*/("""
                    """),format.raw/*278.21*/("""renderSystemPage(true);
                    renderTabs();
                    selectMachine();
                    /* Default GPU to hidden */
                    $("#gpuTable").hide();
                    $("#gpuMemoryChart").hide();
                """),format.raw/*284.17*/("""}"""),format.raw/*284.18*/(""");
        </script>

            <!--Execute periodically (every 2 sec) -->
        <script>
                setInterval(function () """),format.raw/*289.41*/("""{"""),format.raw/*289.42*/("""
                    """),format.raw/*290.21*/("""renderSystemPage(false);
                """),format.raw/*291.17*/("""}"""),format.raw/*291.18*/(""", 2000);
        </script>
            <!--End JavaScript-->

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
object TrainingSystem extends TrainingSystem_Scope0.TrainingSystem
              /*
                  -- GENERATED --
                  DATE: Fri Nov 11 15:48:06 AEDT 2016
                  SOURCE: C:/DL4J/Git/deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-play/src/main/views/org/deeplearning4j/ui/views/training/TrainingSystem.scala.html
                  HASH: f7fc56862b2360dda0de742ea34afdc0e4123a81
                  MATRIX: 600->1|733->39|761->41|884->138|896->142|946->172|2731->1930|2744->1934|2795->1964|2943->2085|2956->2089|3011->2123|3065->2149|3797->2853|3811->2857|3866->2890|4016->3012|4030->3016|4082->3046|4265->3201|4279->3205|4332->3236|4407->3414|4465->3444|4681->3633|4694->3637|4748->3670|6971->5866|6984->5870|7038->5903|7297->6134|7311->6138|7374->6179|7404->6180|8265->7013|8279->7017|8353->7068|8384->7069|8760->7417|8774->7421|8842->7466|8961->7557|8975->7561|9044->7607|9520->8055|9534->8059|9605->8107|9636->8108|10009->8453|10023->8457|10091->8502|10211->8594|10225->8598|10294->8644|10927->9249|10941->9253|11005->9294|11464->9725|11478->9729|11547->9775|11651->9851|11665->9855|11730->9897|11834->9973|11848->9977|11921->10027|12025->10103|12039->10107|12108->10153|12212->10229|12226->10233|12293->10277|12397->10353|12411->10357|12484->10407|14131->12026|14145->12030|14209->12071|14668->12502|14682->12506|14749->12550|14853->12626|14867->12630|14928->12668|15032->12744|15046->12748|15111->12790|15215->12866|15229->12870|15295->12913|15399->12989|15413->12993|15482->13039|15586->13115|15600->13119|15670->13166|15774->13242|15788->13246|15859->13294|17259->14693|17330->14776|17405->14850|17484->14928|17567->15017|17646->15077|17725->15156|17808->15249|17895->15318|17986->15388|18081->15468|18172->15539|18259->15609|18346->15678|18437->15748|18533->15857|18624->15928|18711->15998|18794->16064|18873->16124|18948->16180|19019->16232|19086->16270|21748->18903|21778->18904|21829->18926|22115->19183|22145->19184|22313->19323|22343->19324|22394->19346|22465->19388|22495->19389
                  LINES: 20->1|25->1|26->2|31->7|31->7|31->7|65->41|65->41|65->41|67->43|67->43|67->43|68->44|84->60|84->60|84->60|85->61|85->61|85->61|86->62|86->62|86->62|87->63|88->64|90->66|90->66|90->66|120->96|120->96|120->96|122->98|122->98|122->98|122->98|137->113|137->113|137->113|137->113|141->117|141->117|141->117|142->118|142->118|142->118|149->125|149->125|149->125|149->125|153->129|153->129|153->129|154->130|154->130|154->130|166->142|166->142|166->142|172->148|172->148|172->148|173->149|173->149|173->149|174->150|174->150|174->150|175->151|175->151|175->151|176->152|176->152|176->152|177->153|177->153|177->153|201->177|201->177|201->177|207->183|207->183|207->183|208->184|208->184|208->184|209->185|209->185|209->185|210->186|210->186|210->186|211->187|211->187|211->187|212->188|212->188|212->188|213->189|213->189|213->189|233->209|234->210|235->211|236->212|237->213|238->214|239->215|240->216|241->217|242->218|243->219|244->220|245->221|246->222|247->223|248->224|249->225|250->226|251->227|252->228|253->229|254->230|255->231|301->277|301->277|302->278|308->284|308->284|313->289|313->289|314->290|315->291|315->291
                  -- GENERATED --
              */
          