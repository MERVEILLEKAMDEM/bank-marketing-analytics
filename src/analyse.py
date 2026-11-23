"""Campagnes de télémarketing bancaire : analyse et ciblage sur données réelles.

Données : UCI Bank Marketing (bank-additional-full), 41 188 appels d'une banque
portugaise entre mai 2008 et novembre 2010. Moro, Cortez & Rita (2014),
Decision Support Systems. Licence CC BY 4.0.

Le script produit les graphiques et le tableau de bord HTML de docs/ (publié
avec GitHub Pages) ainsi que results/metrics.json.
"""
import base64
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

RACINE = Path(__file__).resolve().parents[1]
BLEU, ORANGE, GRIS, VERT = "#1f4e79", "#e67e22", "#95a5a6", "#1e8449"
MOIS = ["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
MOIS_FR = ["mars", "avr.", "mai", "juin", "juil.", "août", "sept.", "oct.", "nov.", "déc."]


def charger() -> pd.DataFrame:
    df = pd.read_csv(RACINE / "data" / "bank-full.csv", sep=";")
    df["souscrit"] = (df["y"] == "yes").astype(int)
    return df


def taux(df, col, ordre=None, min_n=200):
    t = df.groupby(col)["souscrit"].agg(["mean", "size"]).rename(columns={"mean": "taux", "size": "n"})
    t = t[t["n"] >= min_n]
    return t.reindex([o for o in ordre if o in t.index]) if ordre else t.sort_values("taux")


def fig_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def barres(t, titre, labels=None, horiz=True, moyenne=None):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    lab = labels or [str(i) for i in t.index]
    if horiz:
        ax.barh(lab, t["taux"] * 100, color=BLEU)
        if moyenne is not None:
            ax.axvline(moyenne * 100, color=ORANGE, ls="--", lw=1)
        ax.set_xlabel("Taux de souscription (%)")
    else:
        ax.bar(lab, t["taux"] * 100, color=BLEU)
        if moyenne is not None:
            ax.axhline(moyenne * 100, color=ORANGE, ls="--", lw=1)
        ax.set_ylabel("Taux de souscription (%)")
    ax.set_title(titre, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def main():
    df = charger()
    moy = df["souscrit"].mean()
    figs, m = {}, {"lignes": len(df), "taux_global": round(moy, 4)}

    # ---------------- Analyse descriptive ------------------------------------
    t_out = taux(df, "poutcome", ["nonexistent", "failure", "success"])
    figs["historique"] = fig_b64(barres(
        t_out, "Résultat de la campagne précédente", ["Jamais contacté", "Échec", "Succès"], moyenne=moy))
    m["taux_si_succes_precedent"] = round(t_out.loc["success", "taux"], 4)

    df["nb_appels"] = df["campaign"].clip(upper=8).astype(str).replace("8", "8+")
    t_c = taux(df, "nb_appels", [str(i) for i in range(1, 8)] + ["8+"], min_n=100)
    figs["fatigue"] = fig_b64(barres(t_c, "Nombre d'appels pendant la campagne", horiz=False, moyenne=moy))
    m["taux_1_appel"] = round(t_c.loc["1", "taux"], 4)
    m["taux_8plus_appels"] = round(t_c.loc["8+", "taux"], 4)

    t_m = taux(df, "month", MOIS, min_n=100)
    figs["mois"] = fig_b64(barres(
        t_m, "Mois de l'appel", [MOIS_FR[MOIS.index(i)] for i in t_m.index], horiz=False, moyenne=moy))

    df["age_cl"] = pd.cut(df["age"], [0, 25, 35, 45, 55, 65, 120],
                          labels=["< 25", "25-34", "35-44", "45-54", "55-64", "65+"])
    t_a = taux(df, "age_cl", ["< 25", "25-34", "35-44", "45-54", "55-64", "65+"], min_n=100)
    figs["age"] = fig_b64(barres(t_a, "Âge du client", horiz=False, moyenne=moy))

    df["euribor_cl"] = pd.cut(df["euribor3m"], [0, 1, 2, 4, 6], labels=["< 1 %", "1-2 %", "2-4 %", "> 4 %"])
    t_e = taux(df, "euribor_cl", ["< 1 %", "1-2 %", "2-4 %", "> 4 %"], min_n=100)
    figs["euribor"] = fig_b64(barres(t_e, "Taux Euribor 3 mois au moment de l'appel", horiz=False, moyenne=moy))

    # ---------------- Modèle de ciblage --------------------------------------
    # 'duration' est exclue : la durée de l'appel n'est connue qu'après l'appel
    # (fuite de données signalée par les auteurs du jeu).
    cible = "souscrit"
    exclues = ["y", "souscrit", "duration", "nb_appels", "age_cl", "euribor_cl"]
    X, y = df.drop(columns=exclues), df[cible]
    cat = X.select_dtypes(exclude="number").columns.tolist()
    num = X.select_dtypes(include="number").columns.tolist()

    # Les lignes sont dans l'ordre chronologique : on entraîne sur le passé
    # (80 % premiers appels) et on teste sur les 20 % les plus récents.
    coupe = int(len(df) * 0.8)
    Xtr, Xte, ytr, yte = X.iloc[:coupe], X.iloc[coupe:], y.iloc[:coupe], y.iloc[coupe:]

    modeles = {
        "Régression logistique": make_pipeline(
            ColumnTransformer([("num", StandardScaler(), num),
                               ("cat", OneHotEncoder(handle_unknown="ignore"), cat)]),
            LogisticRegression(max_iter=2000, class_weight="balanced")),
        "Gradient boosting": make_pipeline(
            ColumnTransformer([("num", "passthrough", num),
                               ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat)]),
            HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0)),
    }
    m["modeles"] = {}
    scores = {}
    for nom, mod in modeles.items():
        mod.fit(Xtr, ytr)
        p = mod.predict_proba(Xte)[:, 1]
        scores[nom] = p
        m["modeles"][nom] = {"roc_auc": round(roc_auc_score(yte, p), 3),
                             "pr_auc": round(average_precision_score(yte, p), 3)}
    meilleur = max(m["modeles"], key=lambda k: m["modeles"][k]["pr_auc"])
    p = scores[meilleur]
    m["meilleur_modele"] = meilleur
    m["taux_test"] = round(yte.mean(), 4)

    # Courbe de gain : part des souscripteurs captés en appelant les x % les mieux notés
    ordre = np.argsort(-p)
    cumul = np.cumsum(yte.values[ordre]) / yte.sum()
    part = np.arange(1, len(p) + 1) / len(p)
    for k in (0.2, 0.3, 0.5):
        m[f"capture_top_{int(k*100)}"] = round(float(cumul[int(k * len(p)) - 1]), 3)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(part * 100, cumul * 100, color=BLEU, lw=2, label=f"{meilleur}")
    ax.plot([0, 100], [0, 100], color=GRIS, ls="--", label="Appels au hasard")
    ax.axvline(30, color=ORANGE, ls=":", lw=1)
    ax.set_xlabel("% de clients appelés (les mieux notés d'abord)")
    ax.set_ylabel("% des souscripteurs atteints")
    ax.set_title("Courbe de gain sur la période la plus récente", fontsize=11)
    ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
    figs["gain"] = fig_b64(fig)

    imp = permutation_importance(modeles[meilleur], Xte, yte, scoring="average_precision",
                                 n_repeats=5, random_state=0)
    s = pd.Series(imp.importances_mean, index=X.columns).sort_values().tail(8)
    noms = {"nr.employed": "Nb de salariés (indicateur macro)", "euribor3m": "Euribor 3 mois",
            "pdays": "Jours depuis le dernier contact", "month": "Mois", "contact": "Canal (fixe/mobile)",
            "poutcome": "Résultat campagne précédente", "cons.conf.idx": "Confiance des ménages",
            "emp.var.rate": "Variation de l'emploi", "cons.price.idx": "Indice des prix",
            "age": "Âge", "campaign": "Nb d'appels", "job": "Métier", "day_of_week": "Jour",
            "previous": "Contacts précédents", "education": "Niveau d'études", "default": "Défaut de crédit"}
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.barh([noms.get(i, i) for i in s.index], s.values, color=VERT)
    ax.set_title("Variables les plus utiles au modèle (importance par permutation)", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    figs["importance"] = fig_b64(fig)
    m["top_variables"] = [noms.get(i, i) for i in s.index[::-1][:4]]

    (RACINE / "results").mkdir(exist_ok=True)
    (RACINE / "results" / "metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    ecrire_dashboard(m, figs)
    print(json.dumps(m, ensure_ascii=False, indent=2))


def ecrire_dashboard(m, figs):
    pct = lambda x: f"{x * 100:.1f} %".replace(".", ",")
    mm = m["modeles"][m["meilleur_modele"]]
    kpis = [
        (f"{m['lignes']:,}".replace(",", " "), "appels analysés (2008-2010)"),
        (pct(m["taux_global"]), "taux de souscription moyen"),
        (pct(m["taux_si_succes_precedent"]), "si la campagne précédente avait réussi"),
        (pct(m["capture_top_30"]), "des souscripteurs atteints en appelant 30 % des clients"),
    ]
    k = "".join(f'<div class="kpi"><b>{v}</b><span>{l}</span></div>' for v, l in kpis)
    img = lambda n, t: f'<figure><img src="data:image/png;base64,{figs[n]}" alt="{t}"></figure>'
    html = f"""<!DOCTYPE html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bank Marketing Analytics – Pascale KAMDEM</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#f4f6f9;color:#1c2833}}
header{{background:{BLEU};color:#fff;padding:28px 6vw}}header h1{{margin:0 0 6px;font-size:26px}}
header p{{margin:0;opacity:.85}}main{{padding:24px 6vw;max-width:1200px;margin:auto}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px;margin-bottom:22px}}
.kpi{{background:#fff;border-radius:10px;padding:16px;box-shadow:0 1px 3px #0001}}
.kpi b{{display:block;font-size:28px;color:{BLEU}}}.kpi span{{font-size:13px;color:#566}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px}}
figure{{margin:0;background:#fff;border-radius:10px;padding:10px;box-shadow:0 1px 3px #0001}}
figure img{{width:100%}}h2{{font-size:18px;margin:28px 0 12px}}
.reco{{background:#fff;border-left:4px solid {ORANGE};padding:14px 18px;border-radius:6px}}
footer{{padding:20px 6vw;font-size:12px;color:#667}}a{{color:{BLEU}}}
</style></head><body>
<header><h1>Campagnes de télémarketing bancaire : qui appeler en priorité ?</h1>
<p>Analyse de données réelles et modèle de ciblage · Python, pandas, scikit-learn</p></header>
<main>
<div class="kpis">{k}</div>
<h2>Ce que disent les données</h2>
<div class="grid">{img('historique','Historique')}{img('fatigue','Fatigue')}{img('mois','Mois')}{img('euribor','Euribor')}{img('age','Âge')}</div>
<h2>Modèle de ciblage</h2>
<p>{m['meilleur_modele']} entraîné sur les 80 % d'appels les plus anciens et testé sur les 20 % les plus récents.
ROC AUC = {str(mm['roc_auc']).replace('.', ',')} · PR AUC = {str(mm['pr_auc']).replace('.', ',')}
(taux de souscription de la période de test : {pct(m['taux_test'])}).
La durée de l'appel est exclue : elle n'est connue qu'après l'appel.</p>
<div class="grid">{img('gain','Courbe de gain')}{img('importance','Importance des variables')}</div>
<h2>Recommandations</h2>
<div class="reco"><ul>
<li>Appeler d'abord les clients les mieux notés : 30 % des appels suffisent pour atteindre {pct(m['capture_top_30'])} des souscripteurs.</li>
<li>Relancer en priorité les clients qui avaient souscrit lors d'une campagne précédente ({pct(m['taux_si_succes_precedent'])} de succès).</li>
<li>Limiter le nombre d'appels par client : le taux passe de {pct(m['taux_1_appel'])} au premier appel à {pct(m['taux_8plus_appels'])} au-delà de 8.</li>
<li>Le contexte économique (Euribor, emploi) pèse lourd : adapter l'intensité des campagnes aux conditions de taux.</li>
</ul></div>
</main>
<footer>Données : UCI Bank Marketing, Moro, Cortez &amp; Rita (2014), <i>Decision Support Systems</i>, licence CC BY 4.0.
Code : <a href="https://github.com/MERVEILLEKAMDEM/bank-marketing-analytics">github.com/MERVEILLEKAMDEM/bank-marketing-analytics</a> · Pascale KAMDEM</footer>
</body></html>"""
    (RACINE / "docs").mkdir(exist_ok=True)
    (RACINE / "docs" / "index.html").write_text(html, encoding="utf-8")
    for n, b in figs.items():
        (RACINE / "docs" / f"{n}.png").write_bytes(base64.b64decode(b))


if __name__ == "__main__":
    main()
