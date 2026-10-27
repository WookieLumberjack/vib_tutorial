"""The Theory notes' links: every page link lands on a section that exists, and "Before this page" only
sends the reader back to pages earlier in the course."""

import re

import pytest

from vib_tutorial.gui.links import COURSE, PAGE_TITLES


def notes() -> dict[str, str]:
    from vib_tutorial.gui import (
        background,
        cms_notes,
        complex_modes,
        continuum,
        coupling_notes,
        damping_models,
        duffing,
        frf_matrix,
        isolation,
        modal_test,
        model_updating,
        oma,
        random_vibration,
        rotor_notes,
        sdof,
        shock,
        signals,
        torsional,
        transient,
    )

    return {
        "sdof_page": sdof.NOTES_HTML,
        "transient_page": transient.NOTES_HTML,
        "damping_page": damping_models.NOTES_HTML,
        "isolation_page": isolation.NOTES_HTML,
        "sim_page": background.BACKGROUND_HTML,
        "frf_page": frf_matrix.THEORY_HTML,
        "complex_page": complex_modes.NOTES_HTML,
        "continuum_page": continuum.NOTES_HTML,
        "random_page": random_vibration.NOTES_HTML,
        "shock_page": shock.NOTES_HTML,
        "coupling_page": coupling_notes.THEORY_HTML,
        "cms_page": cms_notes.THEORY_HTML,
        "signals_page": signals.NOTES_HTML,
        "test_page": modal_test.THEORY_HTML,
        "oma_page": oma.NOTES_HTML,
        "updating_page": model_updating.NOTES_HTML,
        "torsional_page": torsional.NOTES_HTML,
        "duffing_page": duffing.NOTES_HTML,
        "rotor_page": rotor_notes.THEORY_HTML,
    }


ORDER = [name for _, pages in COURSE for name, _, _ in pages]


def test_the_course_lists_every_page_once():
    assert sorted(ORDER) == sorted(PAGE_TITLES) == sorted(notes())


@pytest.mark.parametrize("page", ORDER)
def test_page_links_land_on_a_section(page):
    html = notes()
    anchors = {p: set(re.findall(r'<a name=["\']([^"\']+)["\']', h)) for p, h in html.items()}
    text = html[page]
    for target, anchor in re.findall(r'href="page:([a-z_]+)(?:#([^"]+))?"', text):
        assert target in html, target
        assert not anchor or anchor in anchors[target], f"{page} links to {target}#{anchor}"
    for anchor in re.findall(r'href="#([^"]+)"', text):
        assert anchor in anchors[page], f"{page} links to its own #{anchor}"


@pytest.mark.parametrize("page", ORDER[1:])
def test_before_this_page_points_back(page):
    text = notes()[page]
    line = re.search(r"<p><i>Before this page: (.*?)</i></p>", text)
    assert line, f"{page} has no Before this page line"
    for target in re.findall(r'href="page:([a-z_]+)', line.group(1)):
        assert ORDER.index(target) < ORDER.index(page), f"{page} asks for {target} first"


@pytest.mark.parametrize("page", ORDER)
def test_every_page_ends_with_its_takeaways(page):
    assert '<a name="takeaways"></a>What to take away' in notes()[page]
