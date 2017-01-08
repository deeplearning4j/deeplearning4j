
package org.deeplearning4j.ui.views.html.tsne

import play.twirl.api._
import play.twirl.api.TemplateMagic._


     object Tsne_Scope0 {
import models._
import controllers._
import play.api.i18n._
import views.html._
import play.api.templates.PlayMagic._
import play.api.mvc._
import play.api.data._

class Tsne extends BaseScalaTemplate[play.twirl.api.HtmlFormat.Appendable,Format[play.twirl.api.HtmlFormat.Appendable]](play.twirl.api.HtmlFormat) with play.twirl.api.Template0[play.twirl.api.HtmlFormat.Appendable] {

  /**/
  def apply():play.twirl.api.HtmlFormat.Appendable = {
    _display_ {
      {


Seq[Any](format.raw/*1.1*/("""<!DOCTYPE html>
<html lang="en">
    <head>
        <meta charset="utf-8" />
        <title>T-SNE renders</title>

            <!-- jQuery -->
            <!--
    <script src="//ajax.googleapis.com/ajax/libs/jquery/1/jquery.min.js"></script>
    <script src="/assets/d3.min.js"></script>
    <script src="/assets/render.js"></script>
    <link href='http://fonts.googleapis.com/css?family=Roboto:400,300' rel='stylesheet' type='text/css'>
    <link rel="stylesheet" href="/assets/bootstrap-3.3.4-dist/css/bootstrap.min.css">
    <link rel="stylesheet" href="/assets/bootstrap-3.3.4-dist/css/bootstrap-theme.min.css">
    <link rel="stylesheet" href="/assets/css/simple-sidebar.css">
    <link rel="stylesheet" href="/assets/css/style.css">
    <script src="https://code.jquery.com/jquery-2.1.3.min.js"></script>
    <script src="/assets/jquery-fileupload.js"></script>
    <script src="/assets/bootstrap-3.3.4-dist/js/bootstrap.min.js"></script>
    -->
            <!-- jQuery -->
        <script src="/assets/legacy/jquery-2.2.0.min.js"></script>

        <link href='/assets/legacy/roboto.css' rel='stylesheet' type='text/css'>

            <!-- Latest compiled and minified CSS -->
        <link rel="stylesheet" href="/assets/legacy/bootstrap.min.css" />

            <!-- Optional theme -->
        <link rel="stylesheet" href="/assets/legacy/bootstrap-theme.min.css" />

            <!-- Latest compiled and minified JavaScript -->
        <script src="/assets/legacy/bootstrap.min.js" ></script>


            <!-- d3 -->
        <script src="/assets/legacy/d3.v3.min.js" charset="utf-8"></script>


        <script src="/assets/legacy/jquery-fileupload.js"></script>

            <!-- Booststrap Notify plugin-->
        <script src="/assets/legacy/bootstrap-notify.min.js"></script>

        <script src="/assets/legacy/common.js"></script>

            <!-- dl4j plot setup -->
        <script src="/assets/legacy/renderTsne.js"></script>


        <style>
        .hd """),format.raw/*52.13*/("""{"""),format.raw/*52.14*/("""
            """),format.raw/*53.13*/("""background-color: #000000;
            font-size: 18px;
            color: #FFFFFF;
        """),format.raw/*56.9*/("""}"""),format.raw/*56.10*/("""
        """),format.raw/*57.9*/(""".block """),format.raw/*57.16*/("""{"""),format.raw/*57.17*/("""
            """),format.raw/*58.13*/("""width: 250px;
            height: 350px;
            display: inline-block;
            border: 1px solid #DEDEDE;
            margin-right: 64px;
        """),format.raw/*63.9*/("""}"""),format.raw/*63.10*/("""
        """),format.raw/*64.9*/(""".hd-small """),format.raw/*64.19*/("""{"""),format.raw/*64.20*/("""
            """),format.raw/*65.13*/("""background-color: #000000;
            font-size: 14px;
            color: #FFFFFF;
        """),format.raw/*68.9*/("""}"""),format.raw/*68.10*/("""

        """),format.raw/*70.9*/("""body """),format.raw/*70.14*/("""{"""),format.raw/*70.15*/("""
            """),format.raw/*71.13*/("""font-family: 'Roboto', sans-serif;
            color: #333;
            font-weight: 300;
            font-size: 16px;
        """),format.raw/*75.9*/("""}"""),format.raw/*75.10*/("""
        """),format.raw/*76.9*/("""#wrap """),format.raw/*76.15*/("""{"""),format.raw/*76.16*/("""
            """),format.raw/*77.13*/("""width: 800px;
            margin-left: auto;
            margin-right: auto;
        """),format.raw/*80.9*/("""}"""),format.raw/*80.10*/("""
        """),format.raw/*81.9*/("""#embed """),format.raw/*81.16*/("""{"""),format.raw/*81.17*/("""
            """),format.raw/*82.13*/("""margin-top: 10px;
        """),format.raw/*83.9*/("""}"""),format.raw/*83.10*/("""
        """),format.raw/*84.9*/("""h1 """),format.raw/*84.12*/("""{"""),format.raw/*84.13*/("""
            """),format.raw/*85.13*/("""text-align: center;
            font-weight: normal;
        """),format.raw/*87.9*/("""}"""),format.raw/*87.10*/("""
        """),format.raw/*88.9*/(""".tt """),format.raw/*88.13*/("""{"""),format.raw/*88.14*/("""
            """),format.raw/*89.13*/("""margin-top: 10px;
            background-color: #EEE;
            border-bottom: 1px solid #333;
            padding: 5px;
        """),format.raw/*93.9*/("""}"""),format.raw/*93.10*/("""
        """),format.raw/*94.9*/(""".txth """),format.raw/*94.15*/("""{"""),format.raw/*94.16*/("""
            """),format.raw/*95.13*/("""color: #F55;
        """),format.raw/*96.9*/("""}"""),format.raw/*96.10*/("""
        """),format.raw/*97.9*/(""".cit """),format.raw/*97.14*/("""{"""),format.raw/*97.15*/("""
            """),format.raw/*98.13*/("""font-family: courier;
            padding-left: 20px;
            font-size: 14px;
        """),format.raw/*101.9*/("""}"""),format.raw/*101.10*/("""
        """),format.raw/*102.9*/(""".axis """),format.raw/*102.15*/("""{"""),format.raw/*102.16*/("""

        """),format.raw/*104.9*/("""}"""),format.raw/*104.10*/("""
        """),format.raw/*105.9*/("""/*.axis path,*/
        /*.axis line """),format.raw/*106.22*/("""{"""),format.raw/*106.23*/("""*/
            /*fill: none;*/
            /*stroke: rgba(0,0,0,0.1);*/
            /*shape-rendering: crispEdges;*/
        /*"""),format.raw/*110.11*/("""}"""),format.raw/*110.12*/("""*/
        /*.axis text """),format.raw/*111.22*/("""{"""),format.raw/*111.23*/("""*/
            /*font-family: sans-serif;*/
            /*font-size: 11px;*/
            /*fill: #666;*/
        /*"""),format.raw/*115.11*/("""}"""),format.raw/*115.12*/("""*/
        /*.label """),format.raw/*116.18*/("""{"""),format.raw/*116.19*/("""*/
            /*font-size:14px;*/
            /*fill:rgba(0,0,0,0.5);*/
            /*shape-rendering:auto;*/
        /*"""),format.raw/*120.11*/("""}"""),format.raw/*120.12*/("""*/
        </style>

        <script>
        $(document).ready(function() """),format.raw/*124.38*/("""{"""),format.raw/*124.39*/("""
            """),format.raw/*125.13*/("""$('#filenamebutton').click(function() """),format.raw/*125.51*/("""{"""),format.raw/*125.52*/("""
                """),format.raw/*126.17*/("""document.getElementById('form').reset();
                $('#form').hide();
                var filename = $('#filename').val();
                $('#filename').val('');
                updateFileName(filename);
                drawTsne();
            """),format.raw/*132.13*/("""}"""),format.raw/*132.14*/(""");

            $('#form').fileUpload("""),format.raw/*134.35*/("""{"""),format.raw/*134.36*/("""success : function(data, textStatus, jqXHR)"""),format.raw/*134.79*/("""{"""),format.raw/*134.80*/("""
                """),format.raw/*135.17*/("""var fullPath = document.getElementById('form').value;
                var filename = data['name'];
                if (fullPath) """),format.raw/*137.31*/("""{"""),format.raw/*137.32*/("""
                    """),format.raw/*138.21*/("""var startIndex = (fullPath.indexOf('\\') >= 0 ? fullPath.lastIndexOf('\\') : fullPath.lastIndexOf('/'));
                    var filename = fullPath.substring(startIndex);
                    if (filename.indexOf('\\') === 0 || filename.indexOf('/') === 0) """),format.raw/*140.86*/("""{"""),format.raw/*140.87*/("""
                        """),format.raw/*141.25*/("""filename = filename.substring(1);
                    """),format.raw/*142.21*/("""}"""),format.raw/*142.22*/("""
                """),format.raw/*143.17*/("""}"""),format.raw/*143.18*/("""

                """),format.raw/*145.17*/("""document.getElementById('form').reset();
                //$('#form').hide();

                updateFileName(filename);
                drawTsne();

            """),format.raw/*151.13*/("""}"""),format.raw/*151.14*/(""",error : function(err) """),format.raw/*151.37*/("""{"""),format.raw/*151.38*/("""
                """),format.raw/*152.17*/("""console.log(err);
            """),format.raw/*153.13*/("""}"""),format.raw/*153.14*/("""}"""),format.raw/*153.15*/(""");


            function updateFileName(name) """),format.raw/*156.43*/("""{"""),format.raw/*156.44*/("""
                """),format.raw/*157.17*/("""/*
                 $.ajax("""),format.raw/*158.25*/("""{"""),format.raw/*158.26*/("""
                 """),format.raw/*159.18*/("""url: './api/update',
                 type: 'POST',
                 dataType: 'json',
                 data: JSON.stringify("""),format.raw/*162.39*/("""{"""),format.raw/*162.40*/(""""url" : name"""),format.raw/*162.52*/("""}"""),format.raw/*162.53*/("""),
                 cache: false,
                 success: function(data, textStatus, jqXHR) """),format.raw/*164.61*/("""{"""),format.raw/*164.62*/("""


                 """),format.raw/*167.18*/("""}"""),format.raw/*167.19*/(""",
                 error: function(jqXHR, textStatus, errorThrown) """),format.raw/*168.66*/("""{"""),format.raw/*168.67*/("""
                 """),format.raw/*169.18*/("""// Handle errors here
                 console.log('ERRORS: ' + textStatus);
                 """),format.raw/*171.18*/("""}"""),format.raw/*171.19*/(""",
                 complete: function() """),format.raw/*172.39*/("""{"""),format.raw/*172.40*/("""
                 """),format.raw/*173.18*/("""}"""),format.raw/*173.19*/("""
                 """),format.raw/*174.18*/("""}"""),format.raw/*174.19*/(""");
                 */
            """),format.raw/*176.13*/("""}"""),format.raw/*176.14*/("""


            """),format.raw/*179.13*/("""drawTsne();

        """),format.raw/*181.9*/("""}"""),format.raw/*181.10*/(""") ;

    </script>

    </head>

    <body>
        <table style="width: 100%; padding: 5px;" class="hd">
            <tbody>
                <tr>
                    <td style="width: 48px;"><a href="/"><img src="/assets/legacy/deeplearning4j.img" border="0"/></a></td>
                    <td>DeepLearning4j UI</td>
                    <td style="width: 512px; text-align: right;" class="hd-small">&nbsp; Available sessions:
                        <select class="selectpicker" id="sessionSelect" onchange="selectNewSession()" style="color: #000000; display: inline-block; width: 256px;">
                            <option value="0" selected="selected">Pick a session to track</option>
                        </select>
                    </td>
                    <td style="width: 256px;">&nbsp; <!-- placeholder for future use --></td>
                </tr>
            </tbody>
        </table>

        <br />
        <div style="text-align: center">
            <div id="embed" style="display: inline-block; width: 1024px; height: 700px; border: 1px solid #DEDEDE;"></div>
        </div>
        <br/>
        <br/>
        <div style="text-align:center; width: 100%; position: fixed; bottom: 0px; left: 0px; margin-bottom: 15px;">
            <div style="display: inline-block; margin-right: 48px;">
                <h5>Upload a file to UI server.</h5>
                <form encType="multipart/form-data" action="/api/upload" method="POST" id="form">
                    <div>

                        <input name="file" type="file" style="width:300px; display: inline-block;" /><input type="submit" value="Upload file" style="display: inline-block;"/>

                    </div>
                </form>
            </div>

            <div style="display: inline-block;">
                <h5>If a file is already present on the server, specify the path/name.</h5>
                <div id="filebutton">
                    <input type="text" id="filename"/>
                    <button id="filenamebutton">Submit</button>
                </div>
            </div>
        </div>
    </body>

</html>"""))
      }
    }
  }

  def render(): play.twirl.api.HtmlFormat.Appendable = apply()

  def f:(() => play.twirl.api.HtmlFormat.Appendable) = () => apply()

  def ref: this.type = this

}


}

/**/
object Tsne extends Tsne_Scope0.Tsne
              /*
                  -- GENERATED --
                  DATE: Sun Jan 08 13:27:00 AEDT 2017
                  SOURCE: c:/DL4J/Git/deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-play/src/main/views/org/deeplearning4j/ui/views/tsne/Tsne.scala.html
                  HASH: f23308c76c9f10ff1a9f7a366962de1aa1c97daa
                  MATRIX: 634->0|2694->2032|2723->2033|2765->2047|2887->2142|2916->2143|2953->2153|2988->2160|3017->2161|3059->2175|3246->2335|3275->2336|3312->2346|3350->2356|3379->2357|3421->2371|3543->2466|3572->2467|3611->2479|3644->2484|3673->2485|3715->2499|3873->2630|3902->2631|3939->2641|3973->2647|4002->2648|4044->2662|4159->2750|4188->2751|4225->2761|4260->2768|4289->2769|4331->2783|4385->2810|4414->2811|4451->2821|4482->2824|4511->2825|4553->2839|4643->2902|4672->2903|4709->2913|4741->2917|4770->2918|4812->2932|4974->3067|5003->3068|5040->3078|5074->3084|5103->3085|5145->3099|5194->3121|5223->3122|5260->3132|5293->3137|5322->3138|5364->3152|5486->3246|5516->3247|5554->3257|5589->3263|5619->3264|5659->3276|5689->3277|5727->3287|5794->3325|5824->3326|5984->3457|6014->3458|6068->3483|6098->3484|6246->3603|6276->3604|6326->3625|6356->3626|6510->3751|6540->3752|6648->3831|6678->3832|6721->3846|6788->3884|6818->3885|6865->3903|7151->4160|7181->4161|7250->4201|7280->4202|7352->4245|7382->4246|7429->4264|7589->4395|7619->4396|7670->4418|7958->4677|7988->4678|8043->4704|8127->4759|8157->4760|8204->4778|8234->4779|8283->4799|8480->4967|8510->4968|8562->4991|8592->4992|8639->5010|8699->5041|8729->5042|8759->5043|8838->5093|8868->5094|8915->5112|8972->5140|9002->5141|9050->5160|9207->5288|9237->5289|9278->5301|9308->5302|9433->5398|9463->5399|9515->5422|9545->5423|9642->5491|9672->5492|9720->5511|9845->5607|9875->5608|9945->5649|9975->5650|10023->5669|10053->5670|10101->5689|10131->5690|10197->5727|10227->5728|10274->5746|10325->5769|10355->5770
                  LINES: 25->1|76->52|76->52|77->53|80->56|80->56|81->57|81->57|81->57|82->58|87->63|87->63|88->64|88->64|88->64|89->65|92->68|92->68|94->70|94->70|94->70|95->71|99->75|99->75|100->76|100->76|100->76|101->77|104->80|104->80|105->81|105->81|105->81|106->82|107->83|107->83|108->84|108->84|108->84|109->85|111->87|111->87|112->88|112->88|112->88|113->89|117->93|117->93|118->94|118->94|118->94|119->95|120->96|120->96|121->97|121->97|121->97|122->98|125->101|125->101|126->102|126->102|126->102|128->104|128->104|129->105|130->106|130->106|134->110|134->110|135->111|135->111|139->115|139->115|140->116|140->116|144->120|144->120|148->124|148->124|149->125|149->125|149->125|150->126|156->132|156->132|158->134|158->134|158->134|158->134|159->135|161->137|161->137|162->138|164->140|164->140|165->141|166->142|166->142|167->143|167->143|169->145|175->151|175->151|175->151|175->151|176->152|177->153|177->153|177->153|180->156|180->156|181->157|182->158|182->158|183->159|186->162|186->162|186->162|186->162|188->164|188->164|191->167|191->167|192->168|192->168|193->169|195->171|195->171|196->172|196->172|197->173|197->173|198->174|198->174|200->176|200->176|203->179|205->181|205->181
                  -- GENERATED --
              */
          