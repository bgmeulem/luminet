from pygments.style import Style
from pygments.token import Keyword, Name, Comment, String, Error, Text, \
     Number, Operator, Generic, Whitespace, Punctuation, Other, Literal
from pygments.styles import get_style_by_name

catppuccin_light = get_style_by_name('catppuccin-latte')

class LightStyle(catppuccin_light):
    """
    This style mimics the catppuccin color scheme.
    """

    background_color = "#f8f9fb"