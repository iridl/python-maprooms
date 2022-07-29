from dash import Dash, dcc, html
from dash.dependencies import Output, Input, State, ALL
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import uuid
from collections import OrderedDict

def gen_table(tcs, dfs, data, severity):
    return html.Table(
        [
            gen_head(tcs, dfs),
            gen_body(tcs, data, severity)
        ], className="supertable"
    )

def head_cell(child, tool=None):
    if tool is not None:
        obj_id = "target-" + str(uuid.uuid4())
        return [ html.Div(child, id=obj_id),
                 dbc.Tooltip(tool, target=obj_id, className="tooltiptext") ]
    else:
        return child

def gen_select_header(col, options, value):
    return html.Select(
        [
            html.Option(v, k, selected=k == value)
            for k, v in options.items()
        ],
        id=col
    )

def gen_head(tcs, dfs):
    col_width = 100 / len(tcs)
    return html.Thead(
        [
            html.Tr([
                html.Th(row[col], style={'width': f"{col_width}%"}) for col in tcs.keys()
            ])
            for row in dfs.to_dict(orient="records")
        ] + [
            html.Tr([
                html.Th(head_cell(
                    c['name'] + (f" ({c['units']})" if c.get('units') else ''),
                    c['tooltip']
                )) for c in tcs.values()
            ])
        ]
    )


def gen_body(tcs, data, severity):

    def fmt(col, row):
        f = tcs[col].get('format', lambda x: x)
        return f(row[col])

    class_name = lambda col_name, row: worst_class(col_name, row, severity)

    return html.Tbody([
        html.Tr([
            html.Td(fmt(col, row), className=class_name(col, row)) for col in tcs.keys()
        ])
        for row in data.to_dict(orient="records")
    ])


def worst_class(col_name, row, severity):
    indicator_col_name = f'worst_{col_name}'
    if indicator_col_name in row and row[indicator_col_name] == 1:
        return f'cell-severity-{severity}'
    return ''
