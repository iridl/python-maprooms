import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
import dash_leaflet as dlf


def Number(id, default, min, max):
    """Provides input for a number in a range.

    Auto-generates a dash bootstrap components
    Input for selecting a number from within a specified range.

    Parameters
    ----------
    id : str
        ID used for Dash callbacks.
    default : int
        Default value that is displayed in the input box when user loads the page.
    min : int
        Minimum value the user can select from.
    max : int
        Maximum value the user can select from.
    Returns
    -------
    dbc.Input : component
        dbc Input component with numerical inputs.
    Notes
    -----
    Examples
    --------
    """
    return dbc.Input(id=id, type="number", min=min, max=max, size="sm",
                     className="m-1 d-inline-block w-auto",debounce=True,value=str(default))

def Month(id, default):
    """Provides a selector for month.

    Auto-generates a dash bootstrap components Input for selecting a month.

    Parameters
    ----------
    id : str
        ID used for Dash callbacks.
    default : int
        Default value that is displayed in the input box when user loads the page.
    Returns
    -------
    dbc.Select : component
       dbc Select component with months of the year as options in dropdown.
    """
    return dbc.Select(id=id, value=default, size="sm", className="m-1 d-inline-block w-auto",
                      options=[
                          {"label": "January", "value": "jan"},
                          {"label": "February", "value": "feb"},
                          {"label": "March", "value": "mar"},
                          {"label": "April", "value": "apr"},
                          {"label": "May", "value": "may"},
                          {"label": "June", "value": "jun"},
                          {"label": "July", "value": "jul"},
                          {"label": "August", "value": "aug"},
                          {"label": "September", "value": "sep"},
                          {"label": "October", "value": "oct"},
                          {"label": "November", "value": "nov"},
                          {"label": "December", "value": "dec"},
                      ])

def DateNoYear(id, defaultDay, defaultMonth):
    """Provides a selector for date.

    Auto-generates dash bootstrap components Input and Selector
    for selecting a date as ('day', 'month').

    Parameters
    ----------
    id : str
        ID used for Dash callbacks.
    defaultDay : int
        Default value that is displayed in the input box when user loads the page.
    defaultMonth : str
        Default value that is displayed in the dropdown when user loads the page.
    Returns
    -------
    [dbc.Input, dbc.Select] : list
       List which includes dbc Input component with days of the month,
       and dbc Select component with months of the year as options in dropdown.
    """
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

def Sentence(*elems):
    """Creates sentences with dash components.

    Creates a sentence structure where any part of the sentence can be strings, Inputs, Dropdowns, etc.

    Parameters
    ----------
    elems : list
        A list of elements to be included in constructing the full sentence.
    Returns
    -------
    dbc.Form : component
        A dbc Form which formats all list elements into a sentence
        where the user can interact with Inputs within the sentence.
    Notes
    ------
    Still in development.
    """
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

def Block(title, *body, width="100%"): #width of the block in its container
    """Separates out components in individual Cards

    Auto-generates a formatted block with a card header and body.

    Parameters
    ----------
    title : str
        Title of the card to be displayed.
    body : str, dbc
       Any number of elements which can be of various types to be
       formatted as a sentence within the card body.
    width : str
        html style attribute value to determine width of the card within its parent container.
    Returns
    -------
    dbc.Card : component
       A dbc Card which has a pre-formatted title and body where the body can be any number of elements.
       Default ` width='100%'`.
    """
    return dbc.Card([
        dbc.CardHeader(title),
        dbc.CardBody(body),
    ], className="mb-4 ml-4 mr-4", style={"display": "inline-block", "width": width})

def Options(options,labels=None):
    """ Creates options for definition of different Dash components.

    Creates a dictionary of 'labels' and 'values'
    to be used as options for an element within different Dash components.
    Parameters
    ----------
    options : list
        List of values (str, int, float, etc.) which are the options of values to select from some data.
    labels : list
        List of values (str, int, float, etc.) which are labels representing the data values defined in `options`,
        which do not have to be identical to the values in `options`.
    Returns
    -------
    list of dicts
        A list which holds a dictionary for each `options` value where key 'value' == `options` value,
        and key 'label' == `labels` value if `label` != 'None'.
    Notes
    -----
        The default `labels=None` will use `options` to define both the labels and the values.
        If `labels` is populated with a list, the labels can be different from the data values.
        In this case, the values must still match the actual data values, whereas the labels do not.
        An error will be thrown if the number of elements in `options` list != `labels` list.
    """
    if labels == None:
        return [
            { "label": opt, "value": opt }
            for opt in options
        ]
    else:
        assert len(labels) == len(options), "The number of labels and values are not equal."
        return [
            { "label": label, "value":value }
            for (label,value) in zip(labels,options)
        ]

def Select(id, options, labels=None, init=0):
    """Provides a selector for a list of options.

    Creates a auto-populated dash bootstrap components Select component.
    Parameters
    ----------
    id : str
        ID used for Dash callbacks.
    options : list
        List of values (str, int, float, etc.) which are the options of values to select from some data.
    labels : list
        List of values (str, int, float, etc.) which are labels representing the data values defined in `options`,
        which do not have to be identical to the values in `options`.
    init : int
        Index value which determines which value from the list of `options` will be displayed when user loads page.
    Returns
    -------
    dbc.Select : component
        A dbc dropdown which is auto-populated with 'values' and 'labels' where key 'value' == `options` value,
        and key 'label' == `labels` value if `label` != 'None'.
    Notes
    -----
        The default `labels=None` will use `options` to define both the labels and the values.
        If `labels` is populated with a list, the labels can be different from the data values.
        In this case, the values must still match the actual data values, whereas the labels do not.
        An error will be thrown if the number of elements in `options` list != `labels` list.
    """
    if labels == None:
        opts = [ dict(label=opt, value=opt) for opt in options ]
    else:
        assert len(labels) == len(options), "The number of labels and values are not equal."
        opts = [dict(label=label, value=opt) for (label,opt) in zip(labels,options)]
    return dbc.Select(id=id, value=options[init],
                      className="m-1 d-inline-block w-auto", options=opts)
