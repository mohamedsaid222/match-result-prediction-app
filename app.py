from pathlib import Path
import json
import sys
import argparse
import hashlib

import numpy as np
import pandas as pd
import joblib

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except Exception:
    st = None
    STREAMLIT_AVAILABLE = False

ARTIFACTS_DIR = Path("artifacts")
PIPELINE_PATH = ARTIFACTS_DIR / "pipeline.pkl"

def load_pipeline(path: Path = PIPELINE_PATH):
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"[load_pipeline] Failed: {e}", file=sys.stderr)
        return None

def _stable_softmax_from_values(values_bytes: bytes):
    digest = hashlib.sha256(values_bytes).digest()
    if len(digest) >= 3:
        arr = np.frombuffer(digest[:3], dtype=np.uint8).astype(float)
    else:
        arr = np.frombuffer(digest, dtype=np.uint8).astype(float)
        if arr.size == 0:
            arr = np.array([1.0, 1.0, 1.0], dtype=float)
    logits = arr.astype(float)
    logits = logits - np.max(logits)
    exps = np.exp(logits)
    probs = exps / np.sum(exps)
    if probs.size < 3:
        probs = np.pad(probs, (0, 3 - probs.size), constant_values=(1e-6,))
        probs = probs / probs.sum()
    elif probs.size > 3:
        probs = probs[:3]
        probs = probs / probs.sum()
    return probs

def demo_predict(values: dict):
    s = json.dumps(values, sort_keys=True).encode("utf-8")
    probs = _stable_softmax_from_values(s)
    labels = ["Home Win", "Draw", "Away Win"]
    if probs.size != 3:
        probs = np.asarray(probs)
        probs = np.resize(probs, 3)
        probs = probs / probs.sum()
    pred_idx = int(np.argmax(probs))
    prob_dict = {labels[i]: float(np.round(float(probs[i]), 6)) for i in range(3)}
    return labels[pred_idx], prob_dict

def _map_model_classes_to_labels(classes):
    labels = []
    for c in classes:
        try:
            s = str(c).strip().lower()
        except Exception:
            labels.append(str(c))
            continue
        if s in {"h", "home", "home_win", "home win", "home-win"}:
            labels.append("Home Win")
            continue
        if s in {"a", "away", "away_win", "away win", "away-win"}:
            labels.append("Away Win")
            continue
        if s in {"d", "draw", "tie", "drawn"}:
            labels.append("Draw")
            continue
        try:
            ci = int(c)
            if ci in {0, 1, 2}:
                labels.append({0: "Home Win", 1: "Draw", 2: "Away Win"}[ci])
                continue
        except Exception:
            pass
        labels.append(str(c))
    return labels

def _get_final_estimator(obj):
    try:
        steps = getattr(obj, "named_steps", None)
        if steps:
            for name in reversed(list(steps.keys())):
                est = steps[name]
                if hasattr(est, "predict"):
                    return est
    except Exception:
        pass
    return obj

def predict_with_pipeline(pipe, X: pd.DataFrame):
    if pipe is None:
        raise ValueError("Pipeline is None")
    est = _get_final_estimator(pipe)
    proba = None
    if hasattr(pipe, "predict_proba"):
        try:
            proba = pipe.predict_proba(X)
        except Exception:
            proba = None
    if proba is None and hasattr(est, "predict_proba"):
        try:
            Xt = pipe.transform(X) if hasattr(pipe, "transform") else X
            proba = est.predict_proba(Xt)
        except Exception:
            try:
                proba = est.predict_proba(X)
            except Exception:
                proba = None
    if proba is not None:
        p = np.asarray(proba)
        if p.ndim == 2:
            p = p[0]
        elif p.ndim == 1:
            pass
        else:
            p = p.flatten()
        if p.size == 0:
            raise ValueError("predict_proba returned empty array")
        classes = getattr(est, "classes_", getattr(pipe, "classes_", None))
        if classes is None:
            classes = list(range(p.size))
        labels = _map_model_classes_to_labels(classes)
        if len(labels) != p.size:
            labels = [f"class_{i}" for i in range(p.size)]
        prob_dict = {labels[i]: float(np.round(float(p[i]), 6)) for i in range(p.size)}
        pred_idx = int(np.argmax(p))
        return labels[pred_idx], prob_dict
    else:
        pred = pipe.predict(X)
        if isinstance(pred, (list, tuple, np.ndarray)):
            pred_val = pred[0]
        else:
            pred_val = pred
        labels = _map_model_classes_to_labels([pred_val])
        return labels[0], {labels[0]: 1.0}

def streamlit_app(pipe):
    if not STREAMLIT_AVAILABLE:
        raise RuntimeError("streamlit not available. Use CLI mode or install streamlit.")
    st.set_page_config(page_title="Match Result Prediction", page_icon="⚽", layout="centered")
    st.title("⚽ Match Result Prediction")
    st.caption("Enter match details to predict the result.")
    if pipe is None:
        st.warning("No trained pipeline found. Using demo mode.")
    with st.form("predict_form"):
        st.subheader("Match Details")
        home_team = st.text_input("Home Team", value="Barcelona")
        away_team = st.text_input("Away Team", value="Real Madrid")
        tournament = st.text_input("Tournament", value="La Liga")
        city = st.text_input("City", value="Barcelona")
        country = st.text_input("Country", value="Spain")
        neutral = st.selectbox("Neutral Venue", [0, 1], index=0)
        submitted = st.form_submit_button("Predict Result")
    if submitted:
        values = {"home_team": home_team, "away_team": away_team, "tournament": tournament, "city": city, "country": country, "neutral": int(neutral)}
        X = pd.DataFrame([values])
        if pipe is None:
            pred, prob_dict = demo_predict(values)
        else:
            pred, prob_dict = predict_with_pipeline(pipe, X)
        st.subheader("Prediction Result")
        st.markdown(f"**{pred}**")
        st.subheader("Probabilities")
        dfp = pd.DataFrame({"Outcome": list(prob_dict.keys()), "Probability": list(prob_dict.values())})
        st.dataframe(dfp, hide_index=True)
        st.bar_chart(dfp.set_index("Outcome"))

def cli_mode(pipe, args=None):
    parser = argparse.ArgumentParser(description="Match result prediction (CLI)")
    parser.add_argument("--home-team", default="Barcelona")
    parser.add_argument("--away-team", default="Real Madrid")
    parser.add_argument("--tournament", default="La Liga")
    parser.add_argument("--city", default="Barcelona")
    parser.add_argument("--country", default="Spain")
    parser.add_argument("--neutral", type=int, choices=[0, 1], default=0)
    parser.add_argument("--test", action="store_true")
    parsed = parser.parse_args(args=args)
    if parsed.test:
        _run_tests()
        return
    values = {"home_team": parsed.home_team, "away_team": parsed.away_team, "tournament": parsed.tournament, "city": parsed.city, "country": parsed.country, "neutral": int(parsed.neutral)}
    print("Input:", values)
    if pipe is None:
        pred, prob_dict = demo_predict(values)
    else:
        X = pd.DataFrame([values])
        pred, prob_dict = predict_with_pipeline(pipe, X)
    print(f"\nPredicted result: {pred}")
    print("Probabilities:")
    for k, v in prob_dict.items():
        print(f"  {k}: {v:.6f}")

def _run_tests():
    vals1 = {"home_team": "A", "away_team": "B", "tournament": "X", "city": "C", "country": "Z", "neutral": 0}
    pred1, probs1 = demo_predict(vals1)
    if not isinstance(pred1, str):
        raise AssertionError("demo_predict should return a string label")
    s = sum(probs1.values())
    if abs(s - 1.0) >= 1e-6:
        raise AssertionError(f"demo probabilities must sum to 1 (got {s})")
    if len(probs1) != 3:
        raise AssertionError("demo probabilities must contain exactly 3 outcomes")
    vals2 = dict(vals1)
    vals2['home_team'] = 'Different'
    pred2, probs2 = demo_predict(vals2)
    if set(probs1.keys()) != set(probs2.keys()):
        raise AssertionError("Probability keys must be consistent")
    pred3, probs3 = demo_predict(vals1)
    if pred1 != pred3 or probs1 != probs3:
        raise AssertionError("demo_predict must be deterministic for identical inputs")
    labels = _map_model_classes_to_labels([0, 1, 2])
    if labels != ["Home Win", "Draw", "Away Win"]:
        raise AssertionError(f"Numeric mapping wrong: {labels}")
    print("All tests passed.")

def main():
    pipe = load_pipeline()
    if STREAMLIT_AVAILABLE:
        try:
            streamlit_app(pipe)
            return
        except Exception as e:
            print(f"[main] Streamlit UI failed: {e}", file=sys.stderr)
    cli_mode(pipe, args=None)

if __name__ == "__main__":
    main()

