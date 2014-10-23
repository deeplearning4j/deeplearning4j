"""
Render t-SNE text labels.
Requires PIL (Python Imaging Library) and ImageMagick "convert" command.
"""

import sys
import Image, ImageFont, ImageDraw, ImageChops, string

import os.path
#DEFAULT_FONT=os.path.join(os.path.expanduser('~'), "fonts/Vera.ttf")
DEFAULT_FONT=None

import tempfile

def render(points, filename, width=3000, height=1800, fontfile=DEFAULT_FONT, fontsize=12, margin=0.05, transparency=0.5):
    """
    Render t-SNE text points to an image file.
    points is a list of tuples of the form (title, x, y).
    filename should be a .png, typically.
    margin is the amount of extra whitespace added at the edges.
    transparency is the amount of transparency in the text.
    @warning: Make sure you open the PNG in Gimp, or something that supports alpha channels. Otherwise, it will just look completely black.
    """
    W = width
    H = height

    #im = Image.new("L", (W, H), 255)
    im = Image.new("RGBA", (W, H), (0,0,0))

    # use a bitmap font
    #font = ImageFont.load("/usr/share/fonts/liberation/LiberationSans-Italic.ttf")

    if fontfile is not None:
        assert os.path.exists(fontfile)
        font = ImageFont.truetype(fontfile, fontsize)
    
    #draw = ImageDraw.Draw(im)
    #draw.text((10, 10), "hello", font=font)
    
    minx = 0
    maxx = 0
    miny = 0
    maxy = 0
    for (title, x, y) in points:
        if minx > x: minx = x
        if maxx < x: maxx = x
        if miny > y: miny = y
        if maxy < y: maxy = y

    dx = maxx - minx
    dy = maxy - miny
    assert dx > 0
    assert dy > 0
    minx -= dx * margin
    miny -= dy * margin
    maxx += dx * margin
    maxy += dy * margin


    alpha = Image.new("L", im.size, "black")

    for (idx, pt) in enumerate(points):
        (title, x, y) = pt
    #    print x, minx
    #    print 1. * (x - minx) / (maxx - minx)
    #    print y, miny
    #    print 1. * (y - miny) / (maxy - miny)
        x = 1. * (x - minx) / (maxx - minx) * W
        y = 1. * (y - miny) / (maxy - miny) * H
    #    draw.text((x, y), w, fill=255, font=font)

    # Make a grayscale image of the font, white on black.
        pos = (x, y)
        imtext = Image.new("L", im.size, 0)
        drtext = ImageDraw.Draw(imtext)
        print >> sys.stderr, "Rendering title (#%d):" % idx, repr(title)
        if fontfile is not None:
            drtext.text(pos, title, font=font, fill=(256-256*transparency))
        else:
            drtext.text(pos, title, fill=(256-256*transparency))
#        drtext.text(pos, title, font=font, fill=128)

    # Add the white text to our collected alpha channel. Gray pixels around
    # the edge of the text will eventually become partially transparent
    # pixels in the alpha channel.
    #    alpha = ImageChops.lighter(alpha, imtext)
        alpha = ImageChops.add(alpha, imtext)
            
    # Make a solid color, and add it to the color layer on every pixel
    # that has even a little bit of alpha showing.
    #    solidcolor = Image.new("RGBA", im.size, "#ffffff")
    #    immask = Image.eval(imtext, lambda p: 120 * (int(p != 0)))
    #    im = Image.composite(solidcolor, im, immask)
    #    draw.text((x, y), w, fill=0, font=font)
    
        print >> sys.stderr, "Rendered word #%d" % idx
    #    if idx % 100 == 99:
    #        break
    
    # Add the alpha channel to the image, and save it out.
    im.putalpha(alpha)

    tmpf = tempfile.NamedTemporaryFile(suffix=".png")

    #im.save("transtext.png", "PNG")
    print >> sys.stderr, "Rendering alpha image to file", tmpf.name
    im.save(tmpf.name)

    cmd = "convert %s -background white -flatten %s" % (tmpf.name, filename)
    print >> sys.stderr, "Flattening image", tmpf.name, "to", filename, "using command:", cmd
    os.system(cmd)
