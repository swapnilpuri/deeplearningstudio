"""Native horizontal section tabs shared by all studios."""
from collections.abc import Mapping, Sequence

import streamlit as st


def render_learning_section_navigation(
    sections: Sequence[str] | Mapping[str, str], module_key: str, *, lazy: bool = True,
):
    """Return ordered section ID -> tab containers (labels may differ from IDs).

    Lazy callers must guard each panel with ``if tab.open``. Existing eager
    studios can use ``lazy=False`` to retain cross-panel computation and avoid
    rerunning on tab clicks. Native tabs supply styling and horizontal overflow.
    Lazy-panel inputs must use stable keys and ``persist_state='session'`` so
    returning to a panel does not reset its inputs or invalidate training.
    """
    labels = dict(sections) if isinstance(sections, Mapping) else {s: s for s in sections}
    if not labels or len(labels) != len(sections) or len(set(labels.values())) != len(labels):
        raise ValueError('Sections must have unique IDs and labels and cannot be empty.')
    if not module_key or any(not s or not label for s, label in labels.items()):
        raise ValueError('Module key, section IDs and labels cannot be empty.')
    tabs = st.tabs(list(labels.values()), key=f'{module_key}_section',
                   on_change='rerun' if lazy else 'ignore')
    return dict(zip(labels, tabs))
