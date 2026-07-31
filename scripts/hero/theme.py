"""Design tokens for the NeuroBranch hero animation.

Dark values are copied verbatim from src/styles/_tokens.scss so the showcase
matches the shipping app. Light values are a contrast-adjusted counterpart of
the same hues, since the app has no light token set yet.
"""

from dataclasses import dataclass


def hexc(s: str):
    s = s.lstrip("#")
    return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))


def mix(a, b, t: float):
    """Linear blend a -> b."""
    return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))


@dataclass(frozen=True)
class Theme:
    name: str
    bg_deep: tuple
    bg: tuple
    panel: tuple
    panel_raised: tuple
    surface: tuple
    line: tuple
    line_soft: tuple
    text: tuple
    muted: tuple
    faint: tuple
    green: tuple
    cyan: tuple
    violet: tuple
    # tensor-role port / edge colours
    p_query: tuple
    p_key: tuple
    p_value: tuple
    p_hidden: tuple
    p_image: tuple
    p_rose: tuple
    # syntax
    syn_kw: tuple
    syn_cls: tuple
    syn_fn: tuple
    syn_num: tuple
    syn_str: tuple
    syn_com: tuple
    # chrome
    titlebar: tuple
    grid_dot: tuple
    shadow_alpha: int
    danger: tuple
    glow_strength: float


DARK = Theme(
    name="dark",
    bg_deep=hexc("070a0b"),
    bg=hexc("090c0d"),
    panel=hexc("0f1415"),
    panel_raised=hexc("151b1c"),
    surface=hexc("1a2122"),
    line=hexc("222a29"),
    line_soft=hexc("161d1d"),
    text=hexc("edf3f1"),
    muted=hexc("9aa6a2"),
    faint=hexc("687470"),
    green=hexc("91c7ad"),
    cyan=hexc("91c3cc"),
    violet=hexc("aaa4d6"),
    p_query=hexc("8fb8d8"),
    p_key=hexc("d3b783"),
    p_value=hexc("aaa4d6"),
    p_hidden=hexc("8b9995"),
    p_image=hexc("91c3cc"),
    p_rose=hexc("d5a3b6"),
    syn_kw=hexc("aaa4d6"),
    syn_cls=hexc("91c3cc"),
    syn_fn=hexc("8fb8d8"),
    syn_num=hexc("d3b783"),
    syn_str=hexc("91c7ad"),
    syn_com=hexc("58635f"),
    titlebar=hexc("0c1011"),
    grid_dot=hexc("1b2322"),
    shadow_alpha=150,
    danger=hexc("e08a8a"),
    glow_strength=1.0,
)

LIGHT = Theme(
    name="light",
    bg_deep=hexc("e4eae8"),
    bg=hexc("eef2f1"),
    panel=hexc("f4f7f6"),
    panel_raised=hexc("fafcfb"),
    surface=hexc("ffffff"),
    line=hexc("c9d4d0"),
    line_soft=hexc("e6ecea"),
    text=hexc("18211f"),
    muted=hexc("5b6864"),
    faint=hexc("8a9591"),
    green=hexc("2f8464"),
    cyan=hexc("2f7d8a"),
    violet=hexc("6455a6"),
    p_query=hexc("356d9e"),
    p_key=hexc("9a6d1e"),
    p_value=hexc("6455a6"),
    p_hidden=hexc("5d6a66"),
    p_image=hexc("2f7d8a"),
    p_rose=hexc("a95576"),
    syn_kw=hexc("6455a6"),
    syn_cls=hexc("22707c"),
    syn_fn=hexc("2b5f8c"),
    syn_num=hexc("8c6317"),
    syn_str=hexc("27725a"),
    syn_com=hexc("9aa5a1"),
    titlebar=hexc("eaefed"),
    grid_dot=hexc("dde4e2"),
    shadow_alpha=90,
    danger=hexc("b4453f"),
    glow_strength=0.45,
)
