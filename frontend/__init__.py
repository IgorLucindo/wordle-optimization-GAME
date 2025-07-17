import streamlit.components.v1 as components
import os

_RELEASE = True # Set this to True for deployment!

if not _RELEASE:
    _component_func = components.declare_component(
        "wordle_game",
        url="http://localhost:3001", # For local development with a JS dev server
    )
else:
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "public")
    _component_func = components.declare_component("wordle_game", path=build_dir)

def wordle_game_component(key=None):
    component_value = _component_func(key=key)
    return component_value