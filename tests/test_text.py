import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "text_pipeline")))

from clean_data import clean_text, compute_truth_score

def test_clean_text_removes_special_chars():
    assert clean_text("Hello!!! #FakeNews @123") == "hello fakenews 123"

def test_clean_text_handles_urls():
    assert clean_text("Check this out: https://example.com") == "check this out URL"

def test_compute_truth_score():
    assert compute_truth_score("true") == 1.0
    assert compute_truth_score("pants-fire") == 0.0
    assert compute_truth_score("unknown-label") == 0.5
    
    
    #====Test Exploration tests====#
import pandas as pd
import numpy as np
from data_exploration import make_autopct  # adjust import
import pytest

# Mock small dataset for testing
df_mock = pd.DataFrame({
    "statement": ["This is true", "Totally false statement", "Half true fact"],
    "truth_score": [1.0, 0.2, 0.6]
})

bins = [-0.01, 0.2, 0.4, 0.6, 0.8, 1.0, 1.01]
labels = ['Pants on Fire', 'False', 'Barely True', 'Half True', 'Mostly True', 'True']


def test_make_autopct():
    f = make_autopct([10, 20, 30])
    result = f(50.0)
    assert "50.0%" in result
    assert "(30)" in result

def test_truth_score_binning():
    df_mock["truth_bin"] = pd.cut(df_mock["truth_score"], bins=bins, labels=labels, right=False)
    assert df_mock["truth_bin"].iloc[0] == "True"
    assert df_mock["truth_bin"].iloc[1] == "False"
    assert df_mock["truth_bin"].iloc[2] == "Half True"

def test_statement_length():
    df_mock["statement_length"] = df_mock["statement"].str.split().str.len()
    assert df_mock["statement_length"].tolist() == [3, 3, 3]

def test_percentages_sum_to_100():
    counts = pd.Series([30, 20, 50])
    percentages = (counts / counts.sum() * 100).round(1)
    assert abs(percentages.sum() - 100.0) < 0.1  # account for rounding

def test_combined_columns_present():
    expected_cols = {"statement", "truth_score"}
    assert expected_cols.issubset(set(df_mock.columns))

#====Generate Embeddings==##
import torch
import numpy as np
from transformers import RobertaTokenizer, RobertaModel
from generate_embeddings import get_roberta_model, get_embedding
import pandas as pd

def test_get_roberta_model_outputs():
    tokenizer, model, device = get_roberta_model()
    assert isinstance(tokenizer, RobertaTokenizer)
    assert isinstance(model, RobertaModel)
    assert device.type in ["cuda", "cpu"]

def test_get_embedding_output_shape():
    tokenizer, model, device = get_roberta_model()
    row = {"statement": "COVID-19 vaccines are safe and effective."}
    emb = get_embedding(row, tokenizer, model, device)
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (768,)

#==Train Random Forest Testing==#
import pytest
from train_rf import load_data, train_random_forest, evaluate_model
from sklearn.ensemble import RandomForestRegressor
import numpy as np

def test_load_data_shapes():
    train_x, test_x, val_x, train_y, test_y, val_y = load_data()
    assert len(train_x) == len(train_y)
    assert len(test_x) == len(test_y)
    assert len(val_x) == len(val_y)
    assert train_x.shape[1] == test_x.shape[1] == val_x.shape[1]

def test_train_random_forest_returns_model():
    train_x, _, _, train_y, _, _ = load_data()
    model = train_random_forest(train_x, train_y)
    assert isinstance(model, RandomForestRegressor)
    preds = model.predict(train_x[:5])
    assert len(preds) == 5
    assert isinstance(preds[0], float)

def test_evaluate_model_returns_values():
    train_x, _, _, train_y, _, _ = load_data()
    model = train_random_forest(train_x, train_y)
    preds, mae = evaluate_model(model, train_x, train_y)
    assert isinstance(preds, np.ndarray)
    assert mae >= 0.0


##===Visualizing Results Testing==##
import pandas as pd
import numpy as np
import os
import tempfile
import umap.umap_ as umap
from sklearn.manifold import TSNE

def test_residuals_correct():
    df = pd.DataFrame({
        "Actual Truth Score": [0.8, 0.4],
        "Predicted Truth Score": [0.6, 0.5]
    })
    residuals = df["Actual Truth Score"] - df["Predicted Truth Score"]
    assert np.allclose(residuals.values, [0.2, -0.1])

def test_truth_score_binning():
    df = pd.DataFrame({"Actual Truth Score": [0.1, 0.35, 0.65, 0.9]})
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ["0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"]
    df["Bin"] = pd.cut(df["Actual Truth Score"], bins=bins, labels=labels)
    assert df["Bin"].tolist() == ["0–0.2", "0.2–0.4", "0.6–0.8", "0.8–1.0"]

def test_umap_output_shape():
    X = np.random.rand(100, 10)
    reducer = umap.UMAP(n_components=3, random_state=42)
    reduced = reducer.fit_transform(X)
    assert reduced.shape == (100, 3)

def test_tsne_output_shape():
    X = np.random.rand(50, 10)
    reduced = TSNE(n_components=2, random_state=42).fit_transform(X)
    assert reduced.shape == (50, 2)



