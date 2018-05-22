/**
 * Knob - jQuery Plugin
 * Downward compatible, touchable dial
 *
 * Version: 1.1.2 (22/05/2012)
 * Requires: jQuery v1.7+
 *
 * Copyright (c) 2011 Anthony Terrien
 * Under MIT and GPL licenses:
 *  http://www.opensource.org/licenses/mit-license.php
 *  http://www.gnu.org/licenses/gpl.html
 *
 * Thanks to vor, eskimoblood, spiffistan
 */
$(function () {

    // Dial logic
    var Dial = function (c, opt) {

        var v = null
            ,ctx = c[0].getContext("2d")
            ,PI2 = 2 * Math.PI
            ,mx ,my ,x ,y
            ,self = this;

        this.onChange = function () {};
        this.onCancel = function () {};
        this.onRelease = function () {};

        this.val = function (nv) {
            if (null != nv) {
                opt.stopper && (nv = Math.max(Math.min(nv, opt.max), opt.min));
                v = nv;
                this.onChange(nv);
                if(opt.dynamicDraw) this.dynamicDraw(nv);
                else this.draw(nv);
            } else {
                var b, a;
                b = a = Math.atan2(mx - x, -(my - y - opt.width / 2)) - opt.angleOffset;
                (a < 0) && (b = a + PI2);
                nv = Math.round(b * (opt.max - opt.min) / PI2) + opt.min;
                return (nv > opt.max) ? opt.max : nv;
            }
        };

        this.change = function (nv) {
            opt.stopper && (nv = Math.max(Math.min(nv, opt.max), opt.min));
            this.onChange(nv);
            this.draw(nv);
        };

        this.angle = function (nv) {
            return (nv - opt.min) * PI2 / (opt.max - opt.min);
        };

        this.draw = function (nv) {

            var a = this.angle(nv)                      // Angle
                ,sa = 1.5 * Math.PI + opt.angleOffset   // Previous start angle
                ,sat = sa                               // Start angle
                ,ea = sa + this.angle(v)                // Previous end angle
                ,eat = sat + a                          // End angle
                ,r = opt.width / 2                      // Radius
                ,lw = r * opt.thickness                 // Line width
                ,cgcolor = Dial.getCgColor(opt.cgColor)
                ,tick
                ;

            ctx.clearRect(0, 0, opt.width, opt.width);
            ctx.lineWidth = lw;

            // Hook draw
            if (opt.draw(a, v, opt, ctx)) { return; }

            for (tick = 0; tick < opt.ticks; tick++) {

                ctx.beginPath();

                if (a > (((2 * Math.PI) / opt.ticks) * tick) && opt.tickColorizeValues) {
                    ctx.strokeStyle = opt.fgColor;
                } else {
                    ctx.strokeStyle = opt.tickColor;
                }

                var tick_sa = (((2 * Math.PI) / opt.ticks) * tick) - (0.5 * Math.PI);
                ctx.arc( r, r, r-lw-opt.tickLength, tick_sa, tick_sa+opt.tickWidth , false);
                ctx.stroke();
            }

            opt.cursor
                && (sa = ea - 0.3)
                && (ea = ea + 0.3)
                && (sat = eat - 0.3)
                && (eat = eat + 0.3);

            switch (opt.skin) {

                case 'default' :

                    ctx.beginPath();
                    ctx.strokeStyle = opt.bgColor;
                    ctx.arc(r, r, r - lw / 2, 0, PI2, true);
                    ctx.stroke();

                    if (opt.displayPrevious) {
                        ctx.beginPath();
                        ctx.strokeStyle = (v == nv) ? opt.fgColor : cgcolor;
                        ctx.arc(r, r, r - lw / 2, sa, ea, false);
                        ctx.stroke();
                    }

                    ctx.beginPath();
                    ctx.strokeStyle = opt.fgColor;
                    ctx.arc(r, r, r - lw / 2, sat, eat, false);
                    ctx.stroke();

                    break;

                case 'tron' :

                    if (opt.displayPrevious) {
                        ctx.beginPath();
                        ctx.strokeStyle = (v == nv) ? opt.fgColor : cgcolor;
                        ctx.arc( r, r, r - lw, sa, ea, false);
                        ctx.stroke();
                    }

                    ctx.beginPath();
                    ctx.strokeStyle = opt.fgColor;
                    ctx.arc( r, r, r - lw, sat, eat, false);
                    ctx.stroke();

                    ctx.lineWidth = 2;
                    ctx.beginPath();
                    ctx.strokeStyle = opt.fgColor;
                    ctx.arc( r, r, r - lw + 1 + lw * 2 / 3, 0, 2 * Math.PI, false);
                    ctx.stroke();

                    break;
            }
        };
        
        var dynamicDrawIndex;
        var dynamicDrawInterval;
        this.dynamicDraw = function (nv) {
        	var instanceOfThis = this;
        	dynamicDrawIndex = opt.min;
        	dynamicDrawInterval = setInterval(function() {
        		instanceOfThis.animateDraw(nv);
            }, 20);
        };
        
        this.animateDraw = function () {
        	if(dynamicDrawIndex > v) {
        		clearInterval(dynamicDrawInterval);
        		v = dynamicDrawIndex;
        	} else {
        		this.draw(dynamicDrawIndex);
        		this.change(dynamicDrawIndex);
        		dynamicDrawIndex++;
        	}
        };

        this.capture = function (e) {
            switch (e.type) {
                case 'mousemove' :
                case 'mousedown' :
                    mx = e.pageX;
                    my = e.pageY;
                    break;
                case 'touchmove' :
                case 'touchstart' :
                    mx = e.originalEvent.touches[0].pageX;
                    my = e.originalEvent.touches[0].pageY;
                    break;
            }
            this.change( this.val() );
        };

        this.cancel = function () {
            self.val(v);
            self.onCancel();
        };

        this.startDrag = function (e) {

            var p = c.offset()
                ,$doc = $(document);

            x = p.left + (opt.width / 2);
            y = p.top;

            this.capture(e);

            // Listen mouse and touch events
            $doc.bind(
                    "mousemove.dial touchmove.dial"
                    ,function (e) {
                        self.capture(e);
                    }
                )
                .bind(
                    // Escape
                    "keyup.dial"
                    ,function (e) {
                        if(e.keyCode === 27) {
                            $doc.unbind("mouseup.dial mousemove.dial keyup.dial");
                            self.cancel();
                        }
                    }
                )
                .bind(
                    "mouseup.dial touchend.dial"
                    ,function (e) {
                        $doc.unbind('mousemove.dial touchmove.dial mouseup.dial touchend.dial keyup.dial');
                        self.val(self.val());
                        self.onRelease(v);
                    }
                );
        };
    };

    // Dial static func
    Dial.getCgColor = function (h) {
        h = h.substring(1,7);
        var rgb = [parseInt(h.substring(0,2),16)
                   ,parseInt(h.substring(2,4),16)
                   ,parseInt(h.substring(4,6),16)];
        return "rgba("+rgb[0]+","+rgb[1]+","+rgb[2]+",.5)";
    };

    // jQuery plugin
    $.fn.knob = $.fn.dial = function (gopt) {

        return this.each(

            function () {

                var $this = $(this), opt;

                if ($this.data('dialed')) { return $this; }
                $this.data('dialed', true);

                opt = $.extend(
                    {
                        // Config
                        'min' : $this.data('min') || 0
                        ,'max' : $this.data('max') || 100
                        ,'stopper' : true
                        ,'readOnly' : $this.data('readonly')

                        // UI
                        ,'cursor' : $this.data('cursor')
                        ,'thickness' : $this.data('thickness') || 0.35
                        ,'width' : $this.data('width') || 200
                        ,'displayInput' : $this.data('displayinput') == null || $this.data('displayinput')
                        ,'displayPrevious' : $this.data('displayprevious')
                        ,'fgColor' : $this.data('fgcolor') || '#87CEEB'
                        ,'cgColor' : $this.data('cgcolor') || $this.data('fgcolor') || '#87CEEB'
                        ,'bgColor' : $this.data('bgcolor') || '#EEEEEE'
                        ,'tickColor' : $this.data('tickColor') || $this.data('fgcolor') || '#DDDDDD'
                        ,'ticks' : $this.data('ticks') || 0
                        ,'tickLength' : $this.data('tickLength') || 0
                        ,'tickWidth' : $this.data('tickWidth') || 0.02
                        ,'tickColorizeValues' : $this.data('tickColorizeValues') || true
                        ,'skin' : $this.data('skin') || 'default'
                        ,'angleOffset': degreeToRadians($this.data('angleoffset'))
                        ,'dynamicDraw': $this.data('dynamicdraw') || false

                        // Hooks
                        ,'draw' :
                                /**
                                 * @param int a angle
                                 * @param int v current value
                                 * @param array opt plugin options
                                 * @param context ctx Canvas context 2d
                                 * @return bool true:bypass default draw methode
                                 */
                                function (a, v, opt, ctx) {}
                        ,'change' :
                                /**
                                 * @param int v Current value
                                 */
                                function (v) {}
                        ,'release' :
                                /**
                                 * @param int v Current value
                                 * @param jQuery ipt Input
                                 */
                                function (v, ipt) {}
                    }
                    ,gopt
                );

                var c = $('<canvas width="' + opt.width + '" height="' + opt.width + '"></canvas>')
                    ,wd = $('<div style=width:' + opt.width + 'px;display:inline;"></div>')
                    ,k
                    ,vl = $this.val()
                    ,initStyle = function () {
                        opt.displayInput
                        && $this.css({
                                    'width' : opt.width / 2 + 'px'
                                    ,'position' : 'absolute'
                                    ,'margin-top' : (opt.width * 5 / 14) + 'px'
                                    ,'margin-left' : '-' + (opt.width * 3 / 4) + 'px'
                                    ,'font-size' : (opt.width / 4) + 'px'
                                    ,'border' : 'none'
                                    ,'background' : 'none'
                                    ,'font-family' : 'Arial'
                                    ,'font-weight' : 'bold'
                                    ,'text-align' : 'center'
                                    ,'color' : opt.fgColor
                                    ,'padding' : '0px'
                                    ,'-webkit-appearance': 'none'
                                    })
                        || $this.css({
                                    'width' : '0px'
                                    ,'visibility' : 'hidden'
                                    });
                    };

                // Canvas insert
                $this.wrap(wd).before(c);

                initStyle();

                // Invoke dial logic
                k = new Dial(c, opt);
                vl || (vl = opt.min);
                $this.val(vl);
                k.val(vl);

                k.onRelease = function (v) {
                                            opt.release(v, $this);
                                        };
                k.onChange = function (v) {
                                            $this.val(v);
                                            opt.change(v);
                                         };

                // bind change on input
                $this.bind(
                        'change'
                        ,function (e) {
                            k.val($this.val());
                        }
                    );

                if (!opt.readOnly) {

                    // canvas
                    c.bind(
                                    "mousedown touchstart"
                                    ,function (e) {
                                        e.preventDefault();
                                        k.startDrag(e);
                                    }
                          )
                     .bind(
                                    "mousewheel DOMMouseScroll"
                                    ,mw = function (e) {
                                        e.preventDefault();
                                        var ori = e.originalEvent
                                            ,deltaX = ori.detail || ori.wheelDeltaX
                                            ,deltaY = ori.detail || ori.wheelDeltaY
                                            ,val = parseInt($this.val()) + (deltaX>0 || deltaY>0 ? 1 : deltaX<0 || deltaY<0 ? -1 : 0);
                                        k.val(val);
                                    }
                        );

                    // input
                    var kval, val, to, m = 1, kv = {37:-1, 38:1, 39:1, 40:-1};
                    $this
                        .bind(
                                    "configure"
                                    ,function (e, aconf) {
                                        var kconf;
                                        for (kconf in aconf) { opt[kconf] = aconf[kconf]; }
                                        initStyle();
                                        k.val($this.val());
                                    }
                            )
                        .bind(
                                    "keydown"
                                    ,function (e) {
                                        var kc = e.keyCode;
                                        if (kc >= 96 && kc <= 105) kc -= 48; //numpad
                                        kval = parseInt(String.fromCharCode(kc));

                                        if (isNaN(kval)) {

                                            (kc !== 13)      // enter
                                            && (kc !== 8)    // bs
                                            && (kc !== 9)    // tab
                                            && (kc !== 189)  // -
                                            && e.preventDefault();

                                            // arrows
                                            if ($.inArray(kc,[37,38,39,40]) > -1) {
                                                k.change(parseInt($this.val()) + kv[kc] * m);

                                                // long time keydown speed-up
                                                to = window.setTimeout(
                                                        function () { m < 20 && m++; }
                                                        ,50
                                                        );

                                                e.preventDefault();
                                            }
                                        }
                                    }
                                )
                          .bind(
                                    "keyup"
                                    ,function(e) {
                                        if (isNaN(kval)) {
                                            if (to) {
                                                window.clearTimeout(to);
                                                to = null;
                                                m = 1;
                                                k.val($this.val());
                                                k.onRelease($this.val(), $this);
                                            } else {
                                                // enter
                                                (e.keyCode === 13)
                                                && k.onRelease($this.val(), $this);
                                            }
                                        } else {
                                            // kval postcond
                                            ($this.val() > opt.max && $this.val(opt.max))
                                            || ($this.val() < opt.min && $this.val(opt.min));
                                        }

                                    }
                                )
                           .bind(
                                    "mousewheel DOMMouseScroll"
                                    ,mw
                                );
                } else {
                    $this.attr('readonly', 'readonly');
                }
            }
        ).parent();
    };

    function degreeToRadians (angle) {
            return $.isNumeric(angle) ? angle * Math.PI / 180 : 0;
    }
});