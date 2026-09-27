"""Applying a Theme: Qt's palette, pyqtgraph's defaults, and the plots already on screen.

The widgets' own colours (curves, masses, tables) are redrawn by each widget's
apply_theme(); restyle() handles what they share: plot backgrounds, axes, titles
and legends, and links in labels.
"""

from __future__ import annotations

import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from . import style
from .style import SYSTEM, THEMES, Theme

SETTINGS = ("vib-tutorial", "vib-tutorial")  # QSettings organization, application
Role = QtGui.QPalette.ColorRole
Group = QtGui.QPalette.ColorGroup

_name = style.LIGHT.name  # the theme picked by the user: a THEMES key or SYSTEM


def names() -> list[str]:
    return [SYSTEM, *THEMES]


def current_name() -> str:
    return _name


def saved_name() -> str:
    value = QtCore.QSettings(*SETTINGS).value("theme", SYSTEM)
    return value if value in names() else SYSTEM


def save_name(name: str) -> None:
    QtCore.QSettings(*SETTINGS).setValue("theme", name)


def system_theme() -> Theme:
    """Dark if the desktop asks for a dark colour scheme, else Light."""
    scheme = QtWidgets.QApplication.instance().styleHints().colorScheme()
    return style.DARK if scheme is QtCore.Qt.ColorScheme.Dark else style.LIGHT


def apply(name: str) -> Theme:
    """Make `name` (a THEMES key or SYSTEM) the current theme for new and restyled widgets."""
    global _name
    _name = name
    app = QtWidgets.QApplication.instance()
    hints = app.styleHints()
    if name == SYSTEM:
        hints.unsetColorScheme()  # report (and follow) the desktop's scheme again
        theme = system_theme()
    else:
        theme = THEMES[name]
        # Native widgets (macOS, Windows) draw themselves light or dark from this.
        hints.setColorScheme(QtCore.Qt.ColorScheme.Dark if theme.dark else QtCore.Qt.ColorScheme.Light)
    app.setPalette(palette(theme))
    pg.setConfigOptions(background=theme.background, foreground=theme.foreground)
    style.set_theme(theme)
    return theme


def palette(theme: Theme) -> QtGui.QPalette:
    """The Qt palette for a theme; the platform's own for the Light theme."""
    if theme.window is None:
        return QtGui.QPalette()  # resets to the platform palette
    c = QtGui.QColor
    button = c(theme.button)
    pal = QtGui.QPalette()
    for role, color in (
        (Role.Window, theme.window),
        (Role.WindowText, theme.text),
        (Role.Base, theme.base),
        (Role.AlternateBase, theme.alt_base),
        (Role.ToolTipBase, theme.window),
        (Role.ToolTipText, theme.text),
        (Role.PlaceholderText, theme.muted),
        (Role.Text, theme.text),
        (Role.Button, theme.button),
        (Role.ButtonText, theme.text),
        (Role.BrightText, "#ffffff"),
        (Role.Highlight, theme.highlight),
        (Role.HighlightedText, theme.highlighted_text),
        (Role.Link, theme.link),
        (Role.LinkVisited, theme.link),
        (Role.Accent, theme.highlight),
    ):
        pal.setColor(role, c(color))
    # Bevels and frames (Fusion) are shaded from these.
    pal.setColor(Role.Light, button.lighter(140))
    pal.setColor(Role.Midlight, button.lighter(115))
    pal.setColor(Role.Mid, button.darker(125))
    pal.setColor(Role.Dark, button.darker(160))
    pal.setColor(Role.Shadow, c("#000000"))
    for role in (Role.WindowText, Role.Text, Role.ButtonText, Role.PlaceholderText):
        pal.setColor(Group.Disabled, role, c(theme.disabled))
    pal.setColor(Group.Disabled, Role.Highlight, button.darker(110))
    pal.setColor(Group.Disabled, Role.Button, button)
    return pal


def mute(widget: QtWidgets.QWidget) -> None:
    """Draw a label's text in the palette's secondary-text colour, whatever the theme."""
    widget.setForegroundRole(Role.PlaceholderText)


def restyle(root: QtWidgets.QWidget) -> None:
    """Plot backgrounds, axes, titles and legends, and label links under `root`, in the current theme."""
    for view in root.findChildren(pg.GraphicsView):
        view.setBackground(style.colors.background)
        restyle_items(view.scene().items())
    # A link's colour is fixed when the text is set, from the palette of the time.
    for label in root.findChildren(QtWidgets.QLabel):
        text = label.text()
        if "<a " in text:
            label.clear()  # setText ignores unchanged text
            label.setText(text)


def restyle_items(items) -> None:
    """Axes, titles and legends among `items` (e.g. a scene's, or one PlotItem's children)."""
    fg = style.colors.foreground
    for item in items:
        if isinstance(item, pg.AxisItem):
            item.setPen(fg)
            item.setTextPen(fg)
        elif isinstance(item, pg.LegendItem):
            item.setBrush(style.current().legend_brush())
            item.setLabelTextColor(fg)
            for _, label in item.items:
                label.setText(label.text)
        elif isinstance(item, pg.LabelItem) and item.opts.get("color") is None:
            item.setText(item.text)  # re-rendered in the new default foreground


def restyle_plot_item(plot: pg.PlotItem) -> None:
    """One PlotItem, whether or not it is in a scene right now."""
    items = [plot.getAxis(side) for side in ("left", "bottom", "right", "top")] + [plot.titleLabel]
    if plot.legend is not None:
        items.append(plot.legend)
    restyle_items(items)
