import dash
from dash import html
import dash_bootstrap_components as dbc


def Text(id, default):
    """Provides input for text.

    Auto-generates a dash bootstrap components
    Input for selecting text.

    Parameters
    ----------
    id : str
        ID used for Dash callbacks.
    default : str
        Default value that is displayed in the input box when user loads the page.

    Returns
    -------
    dbc.Input : component
        dbc Input component with text inputs.
    """
    return dbc.Input(
        id=id,
        type="text",
        size="sm",
        class_name="m-0 p-0 d-inline-block w-auto",
        debounce=True,
        value=default,
    )


def Number(id, default=None, min=None, max=None, width="auto"):
    """Provides input for a number in a range.

    Auto-generates a dash bootstrap components
    Input for selecting a number from within a specified range.

    Parameters
    ----------
    id : str
        ID used for Dash callbacks.
    default : float
        Default value that is displayed in the input box when user loads the page.
    min : float, optional
        Minimum value the user can select from. Default is None.
    max : float, optional
        Maximum value the user can select from. Defautl is None.
    width : string, optional
        to control horizontal expansion of control. Accepts CSS entries. Default is "auto"

    Returns
    -------
    dbc.Input : component
        dbc Input component with numerical inputs.
    """
    return dbc.Input(
        id=id,
        type="number",
        min=min,
        max=max,
        class_name="ps-1 pe-0 py-0 d-inline-block",
        debounce=True,
        value=str(default),
        style={"font-size": "10pt", "width": width},
    )


def Month(id, default):
    """Provides a selector for month.

    Auto-generates a dash bootstrap components Input for selecting a month.

    Parameters
    ----------
    id : str
        ID used for Dash callbacks.
    default : str
        Default month value that is displayed in the input box when user loads the page.
        Valid values are the first 3 letters of the month in English with intial in upper case.

    Returns
    -------
    dbc.Select : component
       dbc Select component with months of the year as options in dropdown.
    """
    options = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    labels = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    init = options.index(default)
    return Select(id, options, labels=labels, init=init)


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
        Valid values are the first 3 letters of the month in English with intial in upper case.

    Returns
    -------
    [dbc.Input, dbc.Select] : list
       List which includes dbc Input component with days of the month,
       and dbc Select component with months of the year as options in dropdown.

    See Also
    --------
    Month
    """
    idd = id + "day"
    idm = id + "month"
    return [
        Number(idd, defaultDay, min=1, max=31, width="4em")[0],
        Month(idm, defaultMonth),
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
        groups.append(dbc.Label(elems[i], class_name="m-0"))
        groups.extend(elems[i + 1])

    if tail:
        assert (isinstance(elems[-1], str) or isinstance(elems[-1], html.Span))
        groups.append(dbc.Label(elems[-1], class_name="m-0"))

    return dbc.Form(groups, class_name="py-0 d-inline-block", style={"font-size": "10pt"})

def Block(title, *body, is_on=True, width="auto", border_color="grey", button_id=None):
    """Separates out components in individual Fieldsets

    Auto-generates a formatted block with a fieldset header and body.

    Parameters
    ----------
    title : str
        Title of the fieldset to be displayed.
    body : str, dbc
       Any number of elements which can be of various types to be
       formatted as a sentence within the fieldset body.
    is_on : boolean, optional
       Fieldset is not displayed if False, default is True
    width : str, optional
        html style attribute value to determine width of the fieldset within its
        parent container. Default `width` ="auto".
    button_id : str, optional
        name of id used to replace default Fieldset's Legend with a clickable button
        Displays `title` 

    Returns
    -------
    html.Fieldset : component
       An html Fieldset which has a pre-formatted title and body where the body can
       be any number of elements.
    """
    if is_on:
        the_display = "inline-block"
    else:
        the_display = "none"
    if button_id == None:
        legend = html.Legend(
            title,
            className="position-absolute top-0 start-0 translate-middle-y",
            style={
                "font-size": "10pt",
                "border-style": "outset",
                "border-width": "2px",
                "border-top-width": "0px",
                "border-left-width": "0px",
                "-moz-border-radius": "4px",
                "border-radius": "4px",
                "background-color": "WhiteSmoke",
                "border-color": "LightGrey",
                "padding-bottom": "1px",
                "padding-left": "2px",
                "padding-right": "2px",
                "width": "auto",
            }
        )
    else:
        legend = dbc.Button(
            id=button_id,
            children=title,
            class_name="position-absolute top-0 start-0 translate-middle-y",
            style={
                "font-size": "10pt",
                "border-style": "outset",
                "border-width": "2px",
                "border-top-width": "0px",
                "border-left-width": "0px",
                "-moz-border-radius": "4px",
                "border-radius": "4px",
                "padding-bottom": "1px",
                "padding-left": "2px",
                "padding-right": "2px",
                "width": "auto",
            },
        )
    return dbc.NavItem([html.Fieldset(
        [
            legend,
            html.Div(
                body,
                className="pt-2 mt-0",
                style={
                    "padding-left": "4px",
                    "padding-right": "4px",
                    "padding-bottom": "4px",
                    "margin": "0px",
                    "-moz-border-radius": "8px",
                    "border-radius": "8px",
                    "border-style": "inset",
                    "background-color": "LightGrey",
                    "border-color": border_color,
                },
            ),
        ],
        className="p-0 mt-2 position-relative",
        style={
            "display": the_display,
            "width": width,
            "max-width": "100%",
            "white-space": "nowrap",
        },
    )], class_name="p-0 m-1")

def Options(options, labels=None):
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
    labels : list, optional
        List of values (str, int, float, etc.) which are labels representing the data values defined in `options`,
        which do not have to be identical to the values in `options` , which are the default.
    init : int, optional
        Index value which determines which value from the list of `options` will be displayed when user loads page.
        Default is 0.

    Returns
    -------
    dbc.Select : component
        A dbc dropdown which is auto-populated with 'values' and 'labels' where key 'value' == `options` value,
        and key 'label' == `labels` value if `label` != 'None'.

    Notes
    -----
        If `labels` is populated with a list, the labels can be different from the data values.
        In this case, the values must still match the actual data values, whereas the labels do not.
        An error will be thrown if the number of elements in `options` list != `labels` list.
    """
    if labels == None:
        opts = [ dict(label=opt, value=opt) for opt in options ]
    else:
        assert len(labels) == len(options), "The number of labels and values are not equal."
        opts = [dict(label=label, value=opt) for (label,opt) in zip(labels,options)]
    return dbc.Select(
        id=id,
        value=options[init],
        class_name="d-inline-block w-auto py-0 pl-0 m-0",
        options=opts,
        style={"font-size": "10pt", "max-width": "100%", "white-space": "nowrap"},
    )

def PickPoint(lat_min, lat_max, lat_label, lon_min, lon_max, lon_label):

    return dbc.Row(
        [
            dbc.Col(
                [
                    dbc.Input(
                        id="lat_input",
                        min=lat_min,
                        max=lat_max,
                        type="number",
                        style={"font-size": "10pt", "width": "8em"},
                        class_name="ps-1 pe-0 py-0",
                        placeholder=lat_min,
                    ),
                    dbc.Tooltip(f"{lat_label}", target="lat_input", className="tooltiptext")
                ],
            ),
            dbc.Col(
                [
                    dbc.Input(
                        id="lng_input",
                        min=lon_min,
                        max=lon_max,
                        type="number",
                        style={"font-size": "10pt", "width": "8em"},
                        class_name="ps-1 pe-0 py-0",
                        placeholder=lon_min,
                    ),
                    dbc.Tooltip(f"{lon_label}", target="lng_input", className="tooltiptext")
                ],
            ),
            dbc.Col(dbc.Button(
                id="submit_lat_lng", children="OK", class_name="p-0", style={"font-size": "10pt"}
            )),
        ],
        class_name="g-0",
        justify="start",
    )

