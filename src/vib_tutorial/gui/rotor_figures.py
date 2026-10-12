"""The rotor Theory's figures: SVG drawn from the theme's colours, rendered into the notes' document.

Each figure is built as SVG text from `colors` when the notes are laid out, so
it follows the theme like the plots do. `FigureBrowser` renders them (QtSvg)
for the notes' HTML, which names them as ``<img src="fig:...">``.
"""

from __future__ import annotations

import math

import numpy as np
from PySide6 import QtCore, QtGui, QtSvg, QtWidgets

from ..core.fluid_bearing import attitude
from .style import colors

FONT = 13  # px, the labels' size
SCALE = 2  # rendered at twice the size, for high-density screens


def _text(x: float, y: float, s: str, color: str, anchor: str = "middle", italic: bool = False,
          size: int = FONT) -> str:
    style = " font-style='italic'" if italic else ""
    return (f"<text x='{x:.1f}' y='{y:.1f}' fill='{color}' font-size='{size}' text-anchor='{anchor}'{style}>"
            f"{s}</text>")


def _line(x1: float, y1: float, x2: float, y2: float, color: str, width: float = 1.5, dash: str = "") -> str:
    d = f" stroke-dasharray='{dash}'" if dash else ""
    return (f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' stroke='{color}' "
            f"stroke-width='{width}'{d}/>")


def _head(x: float, y: float, angle: float, color: str, size: float = 9.0) -> str:
    """An arrowhead with its tip at (x, y), pointing along `angle` (radians, SVG axes: y down)."""
    pts = [(x, y)]
    for side in (-1, 1):
        a = angle + math.pi - side * 0.4
        pts.append((x + size * math.cos(a), y + size * math.sin(a)))
    return "<polygon points='" + " ".join(f"{px:.1f},{py:.1f}" for px, py in pts) + f"' fill='{color}'/>"


def _arrow(x1: float, y1: float, x2: float, y2: float, color: str, width: float = 1.5, dash: str = "") -> str:
    return _line(x1, y1, x2, y2, color, width, dash) + _head(x2, y2, math.atan2(y2 - y1, x2 - x1), color)


def _circle(x: float, y: float, r: float, stroke: str = "none", fill: str = "none", width: float = 1.5,
            dash: str = "") -> str:
    d = f" stroke-dasharray='{dash}'" if dash else ""
    return (f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{r:.1f}' stroke='{stroke}' fill='{fill}' "
            f"stroke-width='{width}'{d}/>")


def _point(cx: float, cy: float, r: float, deg: float) -> tuple[float, float]:
    """The point at `deg` (counterclockwise from +x, as on the page) on a circle; SVG's y points down."""
    a = math.radians(deg)
    return cx + r * math.cos(a), cy - r * math.sin(a)


def _arc(cx: float, cy: float, r: float, start: float, end: float, color: str, width: float = 1.5,
         arrow: bool = False) -> str:
    """An arc counterclockwise from `start` to `end` degrees, with an arrowhead at `end` if asked."""
    x1, y1 = _point(cx, cy, r, start)
    x2, y2 = _point(cx, cy, r, end)
    large = 1 if (end - start) % 360 > 180 else 0
    path = (f"<path d='M {x1:.1f} {y1:.1f} A {r:.1f} {r:.1f} 0 {large} 0 {x2:.1f} {y2:.1f}' fill='none' "
            f"stroke='{color}' stroke-width='{width}'/>")
    if arrow:  # the tangent, counterclockwise on the page, is clockwise in SVG's axes
        path += _head(x2, y2, -math.radians(end) - math.pi / 2, color)
    return path


def _svg(width: int, height: int, body: str) -> str:
    return (f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' "
            f"viewBox='0 0 {width} {height}' font-family='sans-serif'>"
            f"<rect x='0.5' y='0.5' width='{width - 1}' height='{height - 1}' fill='{colors.background}' "
            f"stroke='{colors.faint}'/>{body}</svg>")


# ------------------------------------------------------------------ the figures
MARK, LAG = 100.0, 60.0  # degrees: the mark's angle φ (the heavy spot at it), the lag


def spots_svg() -> str:
    """The shaft seen end on: axes, spin, keyphasor mark, heavy spot, high spot and the lag between them."""
    fg, muted, faint = colors.foreground, colors.muted, colors.faint
    shaft, heavy, high = colors.mass[0], colors.force, colors.mass[2]
    ox, oy, whirl, radius = 210.0, 165.0, 34.0, 78.0
    high_deg = MARK - LAG
    cx, cy = _point(ox, oy, whirl, high_deg)
    parts = [
        _arrow(ox - 150, oy, ox + 180, oy, muted, 1.0), _text(ox + 180, oy + 18, "x", fg, italic=True),
        _arrow(ox, oy + 140, ox, oy - 145, muted, 1.0), _text(ox + 12, oy - 135, "y", fg, italic=True),
        _circle(ox, oy, whirl, faint, dash="4 3"),
        _text(ox - 40, oy + 52, "whirl orbit", muted, "end"),
        _circle(cx, cy, radius, shaft, width=2.0),
        _circle(cx, cy, 3, fill=shaft),
        # the spin, round the disc's centre
        _arc(cx, cy, radius + 16, 150, 200, fg, 1.5, arrow=True),
        _text(*_point(cx, cy, radius + 32, 175), "Ω", fg),
        # the angle φ of the mark from +x, at the disc's centre
        _line(cx, cy, cx + radius + 10, cy, faint, 1.0, "4 3"),
        _line(cx, cy, *_point(cx, cy, radius, MARK), faint, 1.0, "4 3"),
        _arc(cx, cy, 22, 0, MARK, muted, 1.2),
        _text(*_point(cx, cy, 32, 20), "φ", fg, italic=True),
    ]
    # The keyphasor mark: a notch on the rim.
    mx, my = _point(cx, cy, radius, MARK)
    a = math.radians(MARK)
    parts.append(_head(mx, my, -a + math.pi, fg, 11))
    parts.append(_text(*_point(cx, cy, radius + 22, MARK - 8), "keyphasor mark", fg, "start"))
    # The heavy spot (at the mark here), and the high spot: the rim's point farthest from the bearings' centre.
    hx, hy = _point(cx, cy, radius - 12, MARK)
    parts += [_circle(hx, hy, 7, fill=heavy), _text(*_point(cx, cy, radius + 14, MARK + 22), "heavy spot", fg, "end")]
    px, py = _point(cx, cy, radius, high_deg)
    parts += [
        _line(ox, oy, px, py, high, 1.2, "5 3"),
        _circle(px, py, 6, high, colors.background, 2.5),
        _text(px + 12, py + 4, "high spot", fg, "start"),
        _arc(cx, cy, radius - 30, high_deg, MARK, heavy, 1.5, arrow=False),
        _text(*_point(cx, cy, radius - 44, (high_deg + MARK) / 2), "lag", fg),
        _text(ox + 8, oy + 16, "O", muted, "start"),
    ]
    return _svg(440, 330, "".join(parts))


def attitude_svg() -> str:
    """A plain journal bearing: the clearance circle, the locus, and the journal at ε and χ."""
    fg, muted, faint = colors.foreground, colors.muted, colors.faint
    bearing, journal, load = colors.structure, colors.mass[0], colors.force
    ox, oy, big, clear = 170.0, 165.0, 125.0, 42.0  # the sleeve's radius and the (exaggerated) clearance
    eps = 0.6
    chi = math.degrees(attitude(eps))
    deg = -90.0 + chi  # from the load line (down), turned with the spin
    jx, jy = _point(ox, oy, eps * clear, deg)
    tx, ty = _point(ox, oy, big, deg)
    parts = [
        _circle(ox, oy, big, bearing, width=2.5),
        _circle(jx, jy, big - clear, journal, width=2.0),
        _arc(jx, jy, big - clear - 18, 200, 250, fg, 1.5, arrow=True),
        _text(*_point(jx, jy, big - clear - 34, 225), "Ω", fg),
        _line(ox, oy - 12, ox, oy + big + 8, faint, 1.0, "4 3"),
        _line(ox, oy, tx, ty, muted, 1.0, "4 3"),
        _circle(ox, oy, clear, faint, dash="3 3"),
    ]
    # The short bearing's locus, ε from 0 (at 90°) to 1 (on the load line).
    pts = []
    for e in np.linspace(0.0, 0.999, 60):
        pts.append(_point(ox, oy, e * clear, -90.0 + math.degrees(attitude(e)) if e else 0.0))
    parts.append("<polyline points='" + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
                 + f"' fill='none' stroke='{colors.mass[2]}' stroke-width='2'/>")
    parts += [
        _arc(ox, oy, 30, -90.0, deg, fg, 1.2),
        _text(*_point(ox, oy, 54, (deg - 90.0) / 2), "χ", fg, italic=True),
        _circle(ox, oy, 2.5, fill=muted), _circle(jx, jy, 3.5, fill=journal),
        _arrow(jx, jy - 70, jx, jy - 5, load, 2.0), _text(jx + 8, jy - 56, "W", fg, "start", italic=True),
        _text(tx + 6, ty + 14, "thinnest film", fg, "start"),
    ]
    # Labels down the right, each with a short leader.
    lx = ox + big + 30
    legend = [
        (oy - 92, colors.structure, "bearing (sleeve)", "solid"),
        (oy - 68, journal, "journal", "solid"),
        (oy - 44, faint, "clearance circle, radius c", "dash"),
        (oy - 20, colors.mass[2], "locus, ε from 0 to 1", "solid"),
    ]
    for y, color, label, kind in legend:
        parts.append(_line(lx, y - 4, lx + 22, y - 4, color, 2.0, "4 3" if kind == "dash" else ""))
        parts.append(_text(lx + 30, y, label.replace("radius c", "radius c<tspan font-size='10' dy='3'>r</tspan>"),
                           fg, "start"))
    parts += [
        _text(lx, oy + 18, f"here ε = {eps} and", muted, "start"),
        _text(lx, oy + 36, f"χ = {chi:.0f}° from the load line", muted, "start"),
    ]
    return _svg(560, 330, "".join(parts))


def campbell_svg() -> str:
    """A schematic Campbell diagram: forward and backward branches, the 1X line, and where they cross."""
    fg, muted, faint = colors.foreground, colors.muted, colors.faint
    forward, backward, hot = colors.mass[0], colors.mass[1], colors.force
    x0, y0, w, h = 60.0, 290.0, 420.0, 250.0
    top = 2.6  # speed and frequency, in units of the natural frequency at rest
    p = 0.6  # I_p/I_d: the forward branch rises more slowly than 1X, so it meets it

    def at(s: float, f: float) -> tuple[float, float]:
        return x0 + w * s / top, y0 - h * f / top

    def curve(fn, color: str, width: float = 2.0, dash: str = "") -> str:
        speeds = np.linspace(0.0, top, 80)
        pts = [at(s, fn(s)) for s in speeds if fn(s) <= top]
        d = f" stroke-dasharray='{dash}'" if dash else ""
        return ("<polyline points='" + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
                + f"' fill='none' stroke='{color}' stroke-width='{width}'{d}/>")

    def branch(sign: float):
        return lambda s: (sign * p * s + math.sqrt((p * s) ** 2 + 4.0)) / 2.0

    parts = [
        _arrow(x0, y0, x0 + w + 20, y0, muted, 1.0), _arrow(x0, y0, x0, y0 - h - 20, muted, 1.0),
        _text(x0 + w + 16, y0 + 20, "spin speed Ω", fg, "end"),
        _text(x0 - 10, y0 - h - 26, "whirl frequency", fg, "start"),
        curve(lambda s: s, fg, 1.5), _text(*at(2.3, 2.42), "1X", fg),
        curve(branch(1.0), forward), _text(*at(2.55, branch(1.0)(2.55) + 0.12), "forward", fg, "end"),
        curve(branch(-1.0), backward), _text(*at(2.55, branch(-1.0)(2.55) - 0.16), "backward", fg, "end"),
        _circle(*at(0, 1), 3, fill=muted), _text(x0 - 8, at(0, 1)[1] + 4, "ω<tspan font-size='10' dy='3'>n</tspan>",
                                                   fg, "end", italic=True),
    ]
    s_f, s_b = 1.0 / math.sqrt(1.0 - p), 1.0 / math.sqrt(1.0 + p)
    fx, fy = at(s_f, s_f)
    bx, by = at(s_b, s_b)
    parts += [
        _line(fx, y0, fx, y0 - 7, muted, 1.5), _circle(fx, fy, 6, fill=hot),
        _text(fx + 12, fy + 24, "critical speed", fg, "start"),
        _circle(bx, by, 6, hot, colors.background, 2.0),
        _text(bx + 12, by + 30, "backward crossing", fg, "start"),
        _text(fx, y0 + 18, "ω<tspan font-size='10' dy='3'>c</tspan>", fg, italic=True),
    ]
    return _svg(540, 330, "".join(parts))


FIGURES = {"fig:spots": spots_svg, "fig:attitude": attitude_svg, "fig:campbell": campbell_svg}


def render(svg: str) -> QtGui.QImage:
    """The SVG as an image at SCALE times its size, marked so it shows at its own size."""
    renderer = QtSvg.QSvgRenderer(QtCore.QByteArray(svg.encode()))
    size = renderer.defaultSize()
    image = QtGui.QImage(size * SCALE, QtGui.QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(image)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    renderer.render(painter)
    painter.end()
    image.setDevicePixelRatio(SCALE)
    return image


class FigureBrowser(QtWidgets.QTextBrowser):
    """A text browser that draws the figures (``<img src="fig:...">``) in the current theme as it lays them out.

    The document caches each image until its HTML is set again, so set it again after a theme change.
    """

    def loadResource(self, kind: int, name: QtCore.QUrl):  # noqa: N802 (Qt's name)
        build = FIGURES.get(name.toString())
        if build is not None and kind == QtGui.QTextDocument.ResourceType.ImageResource:
            return render(build())
        return super().loadResource(kind, name)
