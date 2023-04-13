# take results of 'draug homag rank-queries' invokation
# and displays them for interactive exploration

import draug

import csv
from collections import defaultdict

import streamlit as st


@st.experimental_memo
def load_predictions(path, threshold):

    # kind -> predicted node -> [str]
    ret = {}

    for kind in ("synonym", "parent"):
        name = f"ranking.{kind}.{threshold}.csv"
        with st.spinner(f"loading {name}"):
            with (path / name).open(mode="r") as fd:
                reader = csv.DictReader(fd, delimiter="|")

                ret[kind] = defaultdict(list)
                for d in reader:
                    ret[kind][d["predicted node"]].append(
                        " ".join(
                            (
                                f"**{float(d['score norm']):.3f}**",
                                d["query context"],
                            )
                        )
                    )

    return ret


def load_prediction_paths():
    model = "symptax.v7/bert-base-german-cased"
    model_paths = draug.ENV.DIR.HOMAG_MODELS / model / "models"
    globs = model_paths.glob("*/report/*/vanilla/ranking.synonym.*.csv")

    st.text(f"model: {model}")

    # model checkpoint threshold -> path
    rankings = {}
    with st.spinner("loading prediction selection"):
        for glob in globs:

            threshold = f'0.{glob.stem.split(".")[-1]}'
            model, checkpoint = glob.parts[-5], glob.parts[-3]

            identifier = "|".join([model, checkpoint, threshold])
            rankings[identifier] = glob.parent

    return rankings


def init_sidebar(rankings):
    models = st.sidebar.multiselect("Choose models", rankings.keys(), key="ms_models")
    return models


def populate_column(col, model, path):
    model, checkpoint, threshold = model.split("|")
    st.header(model)
    st.write(f"checkpoint: {checkpoint}, threshold: {threshold}")

    predictions = load_predictions(path, threshold)

    st.sidebar.subheader(model)
    kind = st.sidebar.selectbox("Kind", ("parent", "synonym"), key=f"sb_kind_{model}")

    keys = [
        f"{amount} | {name}"
        for amount, name in sorted(
            [(len(v), k) for k, v in predictions[kind].items()],
            reverse=True,
        )
    ]

    selection = st.sidebar.selectbox("Node", keys, key=f"sb_node_{model}")
    node = selection.split("|")[1].strip()

    if node:
        for pred in predictions[kind][node][: st.session_state.sl_num_preds]:
            st.write(pred)


def main():
    st.header("DRAUG MODEL PREDICTIONS")

    rankings = load_prediction_paths()
    init_sidebar(rankings)

    models = st.session_state.ms_models
    if len(models):
        cols = st.columns(len(models))
        st.sidebar.slider("Predictions", 1, 200, 50, key="sl_num_preds")

        for model, col in zip(models, cols):
            populate_column(col, model, rankings[model])

    else:
        st.write("please select at least 1 model")


main()
