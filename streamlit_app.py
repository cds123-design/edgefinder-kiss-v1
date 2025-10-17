# streamlit_app.py — EdgeFinder (Run-by-sport + Reasoning Cards, mobile-first)
import os
from datetime import datetime
import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="EdgeFinder — Run by Sport", layout="wide")

# ----------------------------- Secrets / Key -----------------------------
def get_api_key():
    try:
        key = (st.secrets.get("ODDS_API_KEY", "") or os.getenv("ODDS_API_KEY", "")).strip()
    except Exception:
        key = (os.getenv("ODDS_API_KEY", "")).strip()
    if not key:
        st.error("Missing ODDS_API_KEY. Add it in Streamlit → Settings → Secrets.")
        st.stop()
    return key

ODDS_KEY = get_api_key()

# ----------------------------- HTTP helper -------------------------------
def odds_get(path, **params):
    url = "https://api.the-odds-api.com" + path
    r = requests.get(url, params={"apiKey": ODDS_KEY, **params}, timeout=25)
    usage = {k: v for k, v in r.headers.items() if k.lower().startswith("x-requests")}
    if usage:
        st.caption("Odds API usage: {}".format(usage))
    if r.status_code != 200:
        try:
            err = r.json()
        except Exception:
            err = {"message": r.text}
        if err.get("error_code") == "OUT_OF_USAGE_CREDITS":
            st.error("The Odds API: out of usage credits. Upgrade/add credits or try later.")
        else:
            st.error("Odds API error {}: {}".format(r.status_code, err.get("message", r.text)[:300]))
        st.stop()
    return r.json()

# ----------------------------- Sports index ------------------------------
FALLBACK_GROUP_FROM_KEY_PREFIX = {
    "basketball_": "Basketball",
    "americanfootball_": "American Football",
    "icehockey_": "Ice Hockey",
    "baseball_": "Baseball",
    "soccer_": "Soccer",
}

@st.cache_data(ttl=12 * 60 * 60, show_spinner=False)
def get_sports_index():
    """
    Returns:
      key_to_group: dict[sport_key] -> "Basketball"/"Soccer"/...
      groups: sorted unique group names
    """
    try:
        data = odds_get("/v4/sports")
        key_to_group = {d["key"]: (d.get("group") or "Other") for d in data}
        groups = sorted({g for g in key_to_group.values()})
        return key_to_group, groups
    except Exception:
        # Fallback: infer by prefix
        def infer_group(k):
            for pref, grp in FALLBACK_GROUP_FROM_KEY_PREFIX.items():
                if k.startswith(pref):
                    return grp
            return "Other"
        # Provide a minimal fallback mapping; keys will be filled from events
        return {}, ["Basketball", "American Football", "Ice Hockey", "Baseball", "Soccer", "Other"]

key_to_group, SPORT_GROUPS = get_sports_index()
default_group = "Basketball" if "Basketball" in SPORT_GROUPS else (SPORT_GROUPS[0] if SPORT_GROUPS else "Basketball")

# ----------------------------- Cached fetchers ---------------------------
@st.cache_data(ttl=120, show_spinner=False)
def fetch_upcoming(regions="us", markets="h2h", odds_format="decimal"):
    return odds_get("/v4/sports/upcoming/odds", regions=regions, markets=markets, oddsFormat=odds_format)

# ----------------------------- Utilities --------------------------------
def fmt_time(iso):
    try:
        t = datetime.fromisoformat((iso or "").replace("Z", "+00:00")).astimezone()
        return t.strftime("%b %d, %I:%M %p")
    except Exception:
        return iso or "TBD"

def parse_home_away(ev):
    home = ev.get("home_team", "TBD")
    away = ev.get("away_team")
    if not away:
        teams = ev.get("teams", [])
        if home and teams:
            away = next((t for t in teams if t != home), teams[0] if teams else "TBD")
        else:
            away = "TBD"
    return home, away

def label_for_event(ev):
    home, away = parse_home_away(ev)
    return "{} @ {} — {}".format(away, home, fmt_time(ev.get("commence_time", "TBD")))

def best_ml_prices(ev, only_book=None):
    """
    Return best (max) moneyline per team across books for 'h2h'.
    If only_book is provided (e.g., "DraftKings"), restrict to that book.
    """
    best = {}
    for bk in ev.get("bookmakers", []):
        title = (bk.get("title") or "").strip()
        if only_book and title.lower() != str(only_book).lower():
            continue
        for m in bk.get("markets", []):
            if m.get("key") != "h2h":
                continue
            for o in m.get("outcomes", []):
                team, price = o.get("name"), o.get("price")
                if team is None or price is None:
                    continue
                if team not in best or price > best[team]["price"]:
                    best[team] = {"price": price, "bookmaker": title or only_book or "Best"}
    return best

def conf_meter(p):
    p = float(p or 0)
    blocks = max(0, min(10, int(round(p * 10))))
    return "■" * blocks + "□" * (10 - blocks)

def implied_from_decimal(dec_a, dec_b):
    """Return vig-normalized implied probs from two decimal lines if both are present."""
    try:
        a = 1.0 / float(dec_a)
        b = 1.0 / float(dec_b)
        s = a + b
        if s <= 0:
            return None, None
        return a / s, b / s
    except Exception:
        return None, None

# ----------------------------- Model hook --------------------------------
try:
    # Expected signature: edgefinder_predict(home, away, sport_key, event) -> {"pick_team": str, "conf": float, "reasons": [str,...]}
    from edgefinder import edgefinder_predict  # noqa: F401
except Exception:
    def edgefinder_predict(home_team, away_team, sport_key, event):
        best = best_ml_prices(event, only_book=st.session_state.get("only_book"))
        ph = best.get(home_team, {}).get("price")
        pa = best.get(away_team, {}).get("price")
        fav = home_team
        if ph is not None and pa is not None:
            fav = home_team if ph <= pa else away_team  # lower decimal price = stronger favorite
            fav_p, _ = implied_from_decimal(ph if fav == home_team else pa, pa if fav == home_team else ph)
            conf = float(fav_p or 0.55)
        else:
            conf = 0.55
        return {"pick_team": fav, "conf": conf, "reasons": []}

# ----------------------------- UI (single screen) ------------------------
st.title("EdgeFinder — Run by Sport")

colA, colB, colC = st.columns([1.4, 1, 1])
with colA:
    run_by_sport = st.toggle("Run by sport (All Upcoming)", value=True)
with colB:
    only_dk = st.toggle("DraftKings only", value=True)
    st.session_state["only_book"] = "DraftKings" if only_dk else None
with colC:
    refresh = st.button("Refresh games", use_container_width=True)

regions, markets, odds_format = "us", "h2h", "decimal"

if run_by_sport:
    selected_group = st.selectbox("Sport", SPORT_GROUPS, index=SPORT_GROUPS.index(default_group) if default_group in SPORT_GROUPS else 0)

# Fetch upcoming on first load or explicit refresh
if "all_upcoming" not in st.session_state or refresh:
    with st.spinner("Fetching upcoming odds…"):
        evs = fetch_upcoming(regions=regions, markets=markets, odds_format=odds_format)
        if not key_to_group:
            keys = {e.get("sport_key", "") for e in evs}
            def infer_group(k):
                for pref, grp in FALLBACK_GROUP_FROM_KEY_PREFIX.items():
                    if k.startswith(pref):
                        return grp
                return "Other"
            for k in keys:
                if k and k not in key_to_group:
                    key_to_group[k] = infer_group(k)
        st.session_state["all_upcoming"] = evs

events = st.session_state.get("all_upcoming", [])
if run_by_sport:
    events = [e for e in events if key_to_group.get(e.get("sport_key", ""), "Other") == selected_group]

if not events:
    st.info("No upcoming events for the current filters.")
else:
    q = st.text_input("Filter by team (optional)")
    filtered = [e for e in events if not q or q.lower() in label_for_event(e).lower()]
    if not filtered:
        st.info("No events match the filter.")
    else:
        labels = [label_for_event(e) for e in filtered]
        ids = list(range(len(filtered)))
        c1, c2 = st.columns([1, 1])
        with c1:
            select_all = st.checkbox("Select all", value=(len(filtered) <= 10))
        selected = ids if select_all else st.multiselect("Select matchup(s)", ids, format_func=lambda i: labels[i])

        if not selected:
            st.warning("Pick at least one matchup (or use Select all).")
        else:
            if st.button("Run Predictions", type="primary", use_container_width=True):
                cards_summary = []
                for i in selected:
                    ev = filtered[i]
                    home, away = parse_home_away(ev)
                    best = best_ml_prices(ev, only_book=("DraftKings" if only_dk else None))
                    dec_home = best.get(home, {}).get("price")
                    dec_away = best.get(away, {}).get("price")

                    sk = ev.get("sport_key")
                    pred = edgefinder_predict(home, away, sk, ev)
                    pick_team = pred.get("pick_team", home)
                    conf = float(pred.get("conf", 0.0))

                    if conf and conf > 0:
                        show_prob = conf
                    elif dec_home and dec_away:
                        ph, pa = implied_from_decimal(dec_home if pick_team == home else dec_away,
                                                      dec_away if pick_team == home else dec_home)
                        show_prob = ph if pick_team == home else pa
                    else:
                        show_prob = 0.55

                    reasons = list(pred.get("reasons", []))
                    if not reasons:
                        if pick_team == home:
                            reasons.append("Home-court advantage favors {}".format(home))
                        else:
                            reasons.append("{} favored despite traveling".format(pick_team))
                        if dec_home and dec_away:
                            fav_price = dec_home if pick_team == home else dec_away
                            dog_price = dec_away if pick_team == home else dec_home
                            reasons.append("Best price gap supports favorite (fav {} vs dog {})".format(fav_price, dog_price))

                    when_txt = fmt_time(ev.get("commence_time"))
                    odds_line = "{} {} | {} {}".format(home, dec_home if dec_home is not None else "—",
                                                       away, dec_away if dec_away is not None else "—")
                    st.markdown(
                        "🏀 **{} vs {}**\n"
                        "───────────────────────────────────────\n"
                        "📅 {}\n"
                        "📈 **Odds (best price):** {}\n"
                        "🤖 **Prediction:** **{}** favored to win ({:.0%})\n"
                        "🟢 **Confidence Meter:** {}\n\n"
                        "**💬 Detailed Reasoning:**".format(
                            away, home, when_txt, odds_line, pick_team, (show_prob or 0), conf_meter(show_prob or 0)
                        )
                    )
                    for r in reasons[:6]:
                        st.markdown(" • {}".format(r))
                    verdict = "Strong" if (show_prob or 0) >= 0.65 else ("Medium" if (show_prob or 0) >= 0.55 else "Lean")
                    st.markdown("───────────────────────────────────────\n✅ **EdgeFinder Model Verdict:** {} edge for **{}**".format(verdict, pick_team))

                    cards_summary.append({
                        "When": when_txt,
                        "Sport": sk,
                        "Matchup": "{} @ {}".format(away, home),
                        "Model Pick": pick_team,
                        "Shown Prob.": "{:.0%}".format((show_prob or 0)),
                        "Best Home ML": dec_home,
                        "Best Away ML": dec_away,
                    })

                if cards_summary:
                    df = pd.DataFrame(cards_summary)
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button("Download CSV (summary)", data=csv,
                                       file_name="edgefinder_cards.csv", mime="text/csv")
