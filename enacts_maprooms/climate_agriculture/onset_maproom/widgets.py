from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash import dash_table

def Number(id, default, min=0, max=5):
    return [ dbc.Input(id=id, type="number", min=min, max=max,
                       size="sm", className="m-1 d-inline-block w-auto",debounce=True,  value=str(default)) ]

def Date(id, defaultDay, defaultMonth):
    return [
        dbc.Input(id=id + "day", type="number", min=1, max=31,
                  size="sm", className="m-1 d-inline-block w-auto", debounce=True, value=str(defaultDay)),
        dbc.Select(id=id + "month", value=defaultMonth, size="sm", className="m-1 d-inline-block w-auto",
                   options=[
                       {"label": "January", "value": "Jan"},
                       {"label": "February", "value": "Feb"},
                       {"label": "March", "value": "Mar"},
                       {"label": "April", "value": "Apr"},
                       {"label": "May", "value": "May"},
                       {"label": "June", "value": "Jun"},
                       {"label": "July", "value": "Jul"},
                       {"label": "August", "value": "Aug"},
                       {"label": "September", "value": "Sep"},
                       {"label": "October", "value": "Oct"},
                       {"label": "November", "value": "Nov"},
                       {"label": "December", "value": "Dec"},
                   ],
        )
    ]

def Units(id):
    return [
        dbc.Select(id=id, value="/percent", size="sm", className="m-1 d-inline-block w-auto",
                   options=[
                       {"label": "fraction", "value": "/unitless"},
                       {"label": "%", "value": "/percent"},
                   ],
        )
    ]

def Sentence(*elems):
    tail = (len(elems) % 2) == 1
    groups = []
    start = 0

    if not isinstance(elems[0], str):
        start = 1
        tail = (len(elems) % 2) == 0
        groups.extend(elems[0])

    for i in range(start, len(elems) - (1 if tail else 0), 2):
        assert (isinstance(elems[i], str) or isinstance(elems[i], html.Span))
        groups.append(dbc.Label(elems[i], size="sm", className="m-1 d-inline-block", width="auto"))
        groups.extend(elems[i + 1])

    if tail:
        assert (isinstance(elems[-1], str) or isinstance(elems[-1], html.Span))
        groups.append(dbc.Label(elems[-1], size="sm", className="m-1 d-inline-block", width="auto"))

    return dbc.Form(groups)


def Block(title, *body, ison=True):
    if ison:
        the_display = "inline-block"
    else:
        the_display = "none"
    return dbc.Card([
        dbc.CardHeader(title),
        dbc.CardBody(body),
    ], className="mb-4", style={"display": the_display})
