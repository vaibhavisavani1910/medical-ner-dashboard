"""Dashboard: Interactive Dash app visualizing medical NER results."""

import json
import sys
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import plotly.express as px
from dash import dash_table, dcc, html

from models import CodedEntity, DashboardData

DATA_PATH = Path("data/coded_entities.json")

CATEGORY_LABELS = {
    "conditions": "Conditions",
    "symptoms": "Symptoms",
    "medications": "Medications",
    "procedures": "Procedures",
}

CATEGORY_COLORS = {
    "conditions": "#EF553B",
    "symptoms": "#AB63FA",
    "medications": "#00CC96",
    "procedures": "#636EFA",
}

TABLE_STYLE = {
    "style_header": {
        "backgroundColor": "#2d3748",
        "color": "#e2e8f0",
        "fontWeight": "bold",
        "border": "1px solid #4a5568",
        "fontSize": "12px",
    },
    "style_cell": {
        "backgroundColor": "#1a202c",
        "color": "#e2e8f0",
        "border": "1px solid #2d3748",
        "fontSize": "12px",
        "padding": "6px 10px",
        "textAlign": "left",
        "whiteSpace": "normal",
        "height": "auto",
        "maxWidth": "250px",
        "overflow": "hidden",
        "textOverflow": "ellipsis",
    },
    "style_data_conditional": [
        {
            "if": {"row_index": "odd"},
            "backgroundColor": "#2d3748",
        }
    ],
}


def load_dashboard_data() -> DashboardData:
    if not DATA_PATH.exists():
        print(
            f"ERROR: {DATA_PATH} not found. Run phase4_codes.py first.",
            file=sys.stderr,
        )
        sys.exit(1)
    with open(DATA_PATH, encoding="utf-8") as fh:
        raw = json.load(fh)
    return DashboardData(**raw)


def make_bar_chart(entities: list[CodedEntity], category: str) -> dcc.Graph:
    names = [e.name.title() for e in entities]
    counts = [e.count for e in entities]
    color = CATEGORY_COLORS[category]

    fig = px.bar(
        x=counts,
        y=names,
        orientation="h",
        labels={"x": "Record Count", "y": ""},
        color_discrete_sequence=[color],
    )
    fig.update_layout(
        paper_bgcolor="#1a202c",
        plot_bgcolor="#1a202c",
        font_color="#e2e8f0",
        margin={"l": 10, "r": 20, "t": 10, "b": 10},
        height=320,
        xaxis={"gridcolor": "#2d3748", "zerolinecolor": "#4a5568"},
        yaxis={"autorange": "reversed", "gridcolor": "#2d3748"},
        showlegend=False,
    )
    fig.update_traces(
        hovertemplate="<b>%{y}</b><br>Count: %{x}<extra></extra>",
        marker_line_width=0,
    )
    return dcc.Graph(figure=fig, config={"displayModeBar": False})


def make_table(entities: list[CodedEntity]) -> dash_table.DataTable:
    rows = [
        {
            "Rank": i + 1,
            "Entity": e.name.title(),
            "Count": e.count,
            "Code": e.code or "—",
            "Code System": e.code_system,
            "Description": e.code_description or "—",
        }
        for i, e in enumerate(entities)
    ]

    columns = [
        {"name": col, "id": col}
        for col in ["Rank", "Entity", "Count", "Code", "Code System", "Description"]
    ]

    return dash_table.DataTable(
        data=rows,
        columns=columns,
        style_header=TABLE_STYLE["style_header"],
        style_cell=TABLE_STYLE["style_cell"],
        style_data_conditional=TABLE_STYLE["style_data_conditional"],
        page_size=10,
        style_table={"overflowX": "auto"},
    )


def build_category_card(category: str, entities: list[CodedEntity]) -> dbc.Card:
    label = CATEGORY_LABELS[category]
    color = CATEGORY_COLORS[category]
    unique_count = len(entities)

    return dbc.Card(
        [
            dbc.CardHeader(
                [
                    html.Div(
                        [
                            html.H5(label, className="mb-0", style={"color": color}),
                            html.Small(
                                f"Top {unique_count} shown",
                                className="text-muted",
                            ),
                        ],
                        className="d-flex justify-content-between align-items-center",
                    )
                ],
                style={"backgroundColor": "#2d3748", "borderColor": "#4a5568"},
            ),
            dbc.CardBody(
                [
                    make_bar_chart(entities, category),
                    html.Hr(style={"borderColor": "#4a5568"}),
                    make_table(entities),
                ],
                style={"backgroundColor": "#1a202c"},
            ),
        ],
        style={"border": f"1px solid {color}33", "borderRadius": "8px"},
        className="mb-4",
    )


def build_summary_badges(data: DashboardData) -> list:
    badges = []
    for cat in ["conditions", "symptoms", "medications", "procedures"]:
        entities: list[CodedEntity] = getattr(data, cat)
        total_mentions = sum(e.count for e in entities)
        coded = sum(1 for e in entities if e.code)
        label = CATEGORY_LABELS[cat]
        color = CATEGORY_COLORS[cat]
        badges.append(
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H4(
                                f"{total_mentions:,}",
                                className="mb-0",
                                style={"color": color},
                            ),
                            html.P(
                                f"Total {label} mentions",
                                className="mb-1 text-muted",
                                style={"fontSize": "12px"},
                            ),
                            html.Small(
                                f"{coded}/{len(entities)} coded",
                                style={"color": "#a0aec0", "fontSize": "11px"},
                            ),
                        ],
                        className="text-center py-2",
                    ),
                    style={
                        "backgroundColor": "#2d3748",
                        "border": f"1px solid {color}55",
                    },
                ),
                md=3,
            )
        )
    return badges


def create_app(data: DashboardData) -> dash.Dash:
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.DARKLY],
        title="Medical NER Dashboard",
    )

    app.layout = dbc.Container(
        [
            # ── Header ────────────────────────────────────────────────────────
            dbc.Row(
                dbc.Col(
                    html.Div(
                        [
                            html.H2(
                                "Medical NER Dashboard",
                                className="mb-0",
                                style={"color": "#e2e8f0", "fontWeight": "700"},
                            ),
                            html.P(
                                "Open Patients Dataset  ·  n = 1,000 clinical records  ·  "
                                "Entities extracted via GPT-4o-mini  ·  "
                                "Codes from NLM ICD-10-CM, RxNorm & SNOMED CT",
                                className="text-muted mb-0",
                                style={"fontSize": "13px"},
                            ),
                        ],
                        className="py-3",
                    )
                )
            ),
            html.Hr(style={"borderColor": "#4a5568"}),
            # ── Summary badges ────────────────────────────────────────────────
            dbc.Row(
                build_summary_badges(data),
                className="mb-4",
            ),
            # ── 2×2 grid ─────────────────────────────────────────────────────
            dbc.Row(
                [
                    dbc.Col(build_category_card("conditions", data.conditions), md=6),
                    dbc.Col(build_category_card("symptoms", data.symptoms), md=6),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(build_category_card("medications", data.medications), md=6),
                    dbc.Col(build_category_card("procedures", data.procedures), md=6),
                ]
            ),
            # ── Footer ────────────────────────────────────────────────────────
            dbc.Row(
                dbc.Col(
                    html.P(
                        "Medical NER Dashboard  ·  Built with Dash + Plotly  ·  "
                        "NER via OpenAI GPT-4o-mini  ·  Codes via NLM APIs",
                        className="text-muted text-center py-3",
                        style={"fontSize": "11px", "borderTop": "1px solid #2d3748"},
                    )
                )
            ),
        ],
        fluid=True,
        style={"backgroundColor": "#171923", "minHeight": "100vh", "padding": "0 24px"},
    )

    return app


if __name__ == "__main__":
    dashboard_data = load_dashboard_data()
    app = create_app(dashboard_data)
    print("Starting Medical NER Dashboard at http://localhost:8050")
    app.run(debug=False, host="0.0.0.0", port=8050)
